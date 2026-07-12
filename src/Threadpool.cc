#include"Threadpool.hpp"

Threadpool::Threadpool(size_t threads){

    for(size_t i = 0; i<threads; i++){
        this->workers.emplace_back([this] {
            while(true){

                std::function<void()> task;

                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);

                    this->cv.wait(lock, [this] {
                        return this->stop || !this->tasks.empty();
                    });
                    
                    if(this->stop && this->tasks.empty()) return;

                    task = std::move(this->tasks.front());

                    this->tasks.pop();
                }

                task();
            }
        });
    }
}

void Threadpool::enqueue_task(std::function<void()> task){
    {
        std::lock_guard<std::mutex> lock(this->queue_mutex);
        if(this->stop) throw std::runtime_error("Enqueueing on stopped Threadpool");
        this->tasks.push(std::move(task));
    }

    this->cv.notify_one();
}

Threadpool::~Threadpool(){
    {
        std::lock_guard<std::mutex> lock(this->queue_mutex);
        this->stop = true;
    }

    this->cv.notify_all();
    for(std::thread& worker : this->workers){
        if(worker.joinable()) worker.join();
    }
}