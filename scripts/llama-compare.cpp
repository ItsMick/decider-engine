// Benchmark client for the public llama API. Not a change to llama.cpp.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;
static double ms(clock_type::time_point start) {
    return std::chrono::duration<double, std::milli>(clock_type::now()-start).count();
}
struct Head {
    std::vector<float> norm, bias, qw, qb, kw, kb;
    int hidden, dim;
    explicit Head(const json & j): hidden(j.at("hidden")), dim(j.at("dim")) {
        norm=j.at("norm.weight").get<std::vector<float>>();
        bias=j.at("norm.bias").get<std::vector<float>>();
        qw=j.at("q.weight").get<std::vector<float>>();
        qb=j.at("q.bias").get<std::vector<float>>();
        kw=j.at("k.weight").get<std::vector<float>>();
        kb=j.at("k.bias").get<std::vector<float>>();
        if(norm.size()!=size_t(hidden)||bias.size()!=norm.size()||qw.size()!=size_t(hidden*dim)||kw.size()!=qw.size()||qb.size()!=size_t(dim)||kb.size()!=qb.size()) throw std::runtime_error("head shape mismatch");
    }
    std::vector<float> project(const float * x, bool query) const {
        if(!x) throw std::runtime_error("missing embedding");
        float mean=0, var=0;
        for(int i=0;i<hidden;i++) mean+=x[i];
        mean/=hidden;
        for(int i=0;i<hidden;i++) var+=(x[i]-mean)*(x[i]-mean);
        float inv=1/std::sqrt(var/hidden+1e-5f);
        std::vector<float> normalized(hidden), out(dim);
        for(int i=0;i<hidden;i++) normalized[i]=(x[i]-mean)*inv*norm[i]+bias[i];
        const auto & w=query?qw:kw; const auto & b=query?qb:kb;
        for(int r=0;r<dim;r++) {
            float sum=0;
            for(int i=0;i<hidden;i++) sum+=w[r*hidden+i]*normalized[i];
            out[r]=sum+b[r];
        }
        return out;
    }
};
struct Capture {
    int hidden;
    std::vector<int> positions;
    std::vector<std::vector<float>> rows;
    static bool eval(ggml_tensor * t, bool ask, void * data) {
        if(std::strcmp(t->name,"result_norm")!=0) return false;
        if(ask) return true;
        auto & self=*static_cast<Capture *>(data);
        if(t->type!=GGML_TYPE_F32||t->ne[0]!=self.hidden||!ggml_is_contiguous(t)) std::abort();
        self.rows.clear();
        for(int p:self.positions) {
            if(p<0||p>=t->ne[1]) std::abort();
            self.rows.emplace_back(self.hidden);
            ggml_backend_tensor_get(t,self.rows.back().data(),size_t(p)*self.hidden*sizeof(float),self.hidden*sizeof(float));
        }
        return true;
    }
};
int main(int argc, char ** argv) try {
    if(argc!=7) throw std::runtime_error("usage: llama-compare MODEL LORA HEAD_JSON CASES_JSON REPEATS THREADS");
    int repeats=std::stoi(argv[5]), threads=std::stoi(argv[6]);
    if(repeats<1||threads<1) throw std::runtime_error("positive repeats/threads required");
    std::ifstream hf(argv[3]), cf(argv[4]); json hj, cases; hf>>hj; cf>>cases;
    Head head(hj);
    Capture capture{head.hidden,{}, {}};
    auto start=clock_type::now();
    llama_backend_init();
    auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto * model=llama_model_load_from_file(argv[1],mp);
    if(!model) throw std::runtime_error("model load failed");
    auto * adapter=llama_adapter_lora_init(model,argv[2]);
    if(!adapter) throw std::runtime_error("adapter load failed");
    auto cp=llama_context_default_params();
    cp.n_ctx=4096; cp.n_batch=1024; cp.n_ubatch=1024;
    cp.n_threads=threads; cp.n_threads_batch=threads;
    // LAST skips the vocabulary head. Capture unpooled hidden rows through the
    // public evaluation callback, before the final last-token pooling operation.
    cp.embeddings=true; cp.pooling_type=LLAMA_POOLING_TYPE_LAST;
    cp.cb_eval=Capture::eval; cp.cb_eval_user_data=&capture;
    cp.type_k=GGML_TYPE_F32; cp.type_v=GGML_TYPE_F32;
    auto * ctx=llama_init_from_model(model,cp);
    if(!ctx) throw std::runtime_error("context init failed");
    float scale=1;
    if(llama_set_adapters_lora(ctx,&adapter,1,&scale)!=0) throw std::runtime_error("adapter activation failed");
    if(llama_model_n_embd(model)!=head.hidden) throw std::runtime_error("hidden shape mismatch");
    std::cout<<json{{"load_ms",ms(start)},{"threads",threads}}<<std::endl;
    for(const auto & c:cases) {
        auto ids=c.at("encoding").at("input_ids").get<std::vector<int>>();
        auto pos=c.at("encoding").at("option_positions").get<std::vector<int>>();
        int answer=c.at("encoding").at("answer_position");
        if(ids.empty()||ids.size()>1024||answer!=int(ids.size())-1||pos.empty()) throw std::runtime_error("invalid encoding");
        capture.positions=pos; capture.positions.push_back(answer);
        auto batch=llama_batch_init(ids.size(),0,1); batch.n_tokens=ids.size();
        for(size_t i=0;i<ids.size();i++) {
            batch.token[i]=ids[i]; batch.pos[i]=i; batch.n_seq_id[i]=1; batch.seq_id[i][0]=0; batch.logits[i]=1;
        }
        for(int p:pos) { if(p<0||p>=answer) throw std::runtime_error("invalid option position"); batch.logits[p]=1; }
        batch.logits[answer]=1;
        for(int sample=0;sample<=repeats;sample++) {
            start=clock_type::now();
            capture.rows.clear();
            llama_memory_clear(llama_get_memory(ctx),true);
            if(llama_decode(ctx,batch)!=0) throw std::runtime_error("decode failed");
            if(capture.rows.size()!=capture.positions.size()) throw std::runtime_error("hidden capture failed");
            auto q=head.project(capture.rows.back().data(),true);
            std::vector<float> logits;
            for(size_t p=0;p<pos.size();p++) {
                auto k=head.project(capture.rows[p].data(),false);
                float dot=0; for(int i=0;i<head.dim;i++) dot+=q[i]*k[i];
                logits.push_back(dot/std::sqrt(float(head.dim)));
            }
            std::cout<<json{{"id",c.at("id")},{"sample",sample},{"warmup",sample==0},{"tokens",ids.size()},{"elapsed_ms",ms(start)},{"logits",logits}}<<std::endl;
        }
        llama_batch_free(batch);
    }
    llama_free(ctx); llama_adapter_lora_free(adapter); llama_model_free(model); llama_backend_free();
    return 0;
} catch(const std::exception & e) { std::cerr<<e.what()<<std::endl; return 1; }
