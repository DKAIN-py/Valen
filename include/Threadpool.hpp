#pragma once

#include<vector>
#include<thread>
#include<mutex>
#include<condition_variable>
#include<queue>
#include<functional>
#include<future>

class Threadpool{
    
private:
    mutable std::vector<std::thread> workers;
    mutable std::queue<std::function<void()>> tasks;
    mutable std::mutex queue_mutex;
    mutable std::condition_variable cv;
    bool stop = false;

public:
    Threadpool(size_t threads);

    void enqueue_task(std::function<void()> task) const;

    ~Threadpool();
};