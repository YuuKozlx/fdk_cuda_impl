#pragma once
// Optional, process-wide timing/watchdog. No CUDA or VTK dependency.
// Enable with YK_TIMING_WATCHDOG=1; YK_TIMING_LOG chooses a persistent log file.
#include <atomic>
#include <algorithm>
#include <stdexcept>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
namespace YK::diagnostics {
inline bool timingEnabled(){static const bool enabled=[](){const char* value=std::getenv("YK_TIMING_WATCHDOG");return value&&std::string(value)!="0";}();return enabled;}
inline double timingOption(const char* name,double fallback){try{const char* value=std::getenv(name);double parsed=value?std::stod(value):fallback;return parsed>0?parsed:fallback;}catch(...){return fallback;}}
class TimingRegistry {
    using Clock=std::chrono::steady_clock;
    struct Active {std::string name,context;Clock::time_point start,last_warning;std::thread::id thread;};
    struct Aggregate {size_t count=0;double total=0,maximum=0;};
    std::mutex mutex_;std::condition_variable wake_;std::map<size_t,Active> active_;std::map<std::string,Aggregate> aggregates_;
    std::ofstream file_;std::thread worker_;bool stop_=false;size_t next_id_=0;
    double slow_ms_=timingOption("YK_TIMING_SLOW_MS",200),heartbeat_ms_=timingOption("YK_TIMING_HEARTBEAT_MS",5000);
    bool detail_=std::getenv("YK_TIMING_DETAIL")!=nullptr;
    static double milliseconds(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
    void emit(const std::string& message){auto epoch=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        file_<<"epoch_ms="<<epoch<<" "<<message<<'\n';file_.flush();}
    void summary(){for(const auto& [name,a]:aggregates_){std::ostringstream out;out<<"kind=summary name="<<std::quoted(name)<<" calls="<<a.count<<" total_ms="<<a.total<<" mean_ms="<<(a.count?a.total/a.count:0)<<" max_ms="<<a.maximum;emit(out.str());}}
    void watch(){std::unique_lock<std::mutex> lock(mutex_);auto last_summary=Clock::now();
        while(!stop_){wake_.wait_for(lock,std::chrono::milliseconds(200),[&]{return stop_;});if(stop_)break;
            for(auto& [id,a]:active_){double elapsed=milliseconds(a.start);if(elapsed>=heartbeat_ms_&&milliseconds(a.last_warning)>=heartbeat_ms_){
                std::ostringstream out;out<<"kind=watchdog id="<<id<<" name="<<std::quoted(a.name)<<" context="<<std::quoted(a.context)<<" elapsed_ms="<<elapsed;
                auto leaf=active_.rbegin();while(leaf!=active_.rend()&&leaf->second.thread!=a.thread)++leaf;
                if(leaf!=active_.rend())out<<" leaf="<<std::quoted(leaf->second.name)<<" leaf_context="<<std::quoted(leaf->second.context);
                emit(out.str());std::clog<<"[timing watchdog] "<<out.str()<<std::endl;a.last_warning=Clock::now();}}
            if(milliseconds(last_summary)>=10000){summary();last_summary=Clock::now();}
        }
    }
public:
    TimingRegistry(){const char* path=std::getenv("YK_TIMING_LOG");std::filesystem::path target=path?path:"fdk-timing.log";
        if(!target.parent_path().empty())std::filesystem::create_directories(target.parent_path());file_.open(target,std::ios::app);
        if(!file_)throw std::runtime_error("cannot create timing log: "+target.string());
        emit("kind=start gpu_substeps_sync=1");worker_=std::thread([this]{try{watch();}catch(...){}});}
    ~TimingRegistry(){{std::lock_guard<std::mutex> lock(mutex_);stop_=true;}wake_.notify_all();if(worker_.joinable())worker_.join();std::lock_guard<std::mutex> lock(mutex_);summary();emit("kind=stop");}
    static TimingRegistry& instance(){static TimingRegistry registry;return registry;}
    size_t begin(std::string name,std::string context){std::lock_guard<std::mutex> lock(mutex_);size_t id=++next_id_;auto now=Clock::now();active_.emplace(id,Active{std::move(name),std::move(context),now,now,std::this_thread::get_id()});return id;}
    void end(size_t id)noexcept{try{std::lock_guard<std::mutex> lock(mutex_);auto it=active_.find(id);if(it==active_.end())return;
        double elapsed=milliseconds(it->second.start);auto& aggregate=aggregates_[it->second.name];++aggregate.count;aggregate.total+=elapsed;aggregate.maximum=std::max(aggregate.maximum,elapsed);
        if(detail_||elapsed>=slow_ms_||aggregate.count==1){std::ostringstream out;out<<"kind=complete id="<<id<<" name="<<std::quoted(it->second.name)<<" context="<<std::quoted(it->second.context)<<" elapsed_ms="<<elapsed;emit(out.str());}
        active_.erase(it);
    }catch(...) {}}
};
class TimingSpan {
    size_t id_=0;
public:
    explicit TimingSpan(std::string name,std::string context={})noexcept{
        static std::atomic<bool> failed{false};if(timingEnabled()&&!failed.load())try{id_=TimingRegistry::instance().begin(std::move(name),std::move(context));}
        catch(const std::exception& error){failed=true;std::clog<<"[timing disabled] "<<error.what()<<std::endl;}catch(...){failed=true;}
    }
    ~TimingSpan(){if(id_)TimingRegistry::instance().end(id_);}
    TimingSpan(const TimingSpan&)=delete;TimingSpan& operator=(const TimingSpan&)=delete;
};
}
