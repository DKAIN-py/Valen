# Valen Inference Engine Architecture Specification

Valen is a high-performance, zero-dependency C++20 inference engine designed to execute compiled deep learning models trained within the **NexusDL** ecosystem. Valen is engineered with a strict **"No-Graph, No-Autograd"** execution paradigm, focusing entirely on bare-metal CPU throughput, flat contiguous memory layouts, and low-latency thread-pool data parallelism.


## 1. Memory Model & Primitive Data Structure (`Nexus`)

The fundamental tensor layout unit in Valen is the **Nexus** class. To eliminate the pointer-chasing latency common in traditional frameworks, Valen strips away multi-dimensional heap arrays and forces an all-contiguous data structure.

### Tensor Attributes

* **Shape (`std::vector<int>`):** A vector tracking structural dimensions. The first element represents the `batch_size`, and the subsequent elements represent the shape matrix order ($M \times N$).
* **Data (`std::vector<float>`):** A single flat 1D heap array storing 32-bit floating-point numbers using **Row-Major Memory Alignment**.

```text
Nested Matrix POV (User Input):          Contiguous Stride POV (Valen Memory):
     ┌───────────────┐                  ┌───────┬───────┬───────┬───────┐
Row 0│  5.5   2.4    │                  │  5.5  │  2.4  │  5.7  │  2.6  │
Row 1│  5.7   2.6    │ ───► Flatten ───►└───────┴───────┴───────┴───────┘
     └───────────────┘                  0       1       2       3
                                        ▲ Row-Major Stride (Index = r * Width + c)

```

### Data Ingestion Primitives

1. **`load_from_binary`:** Reads flat binary streams directly into pre-allocated memory allocations from exported model parameter blocks (`.bin`), matching files to buffers in a single operation.

2. **`load_input`:** Accepts a nested user-facing multi-dimensional structure (`std::vector<std::vector<float>>`), infers shape configurations dynamically, purges initialization layout spaces, and packs elements sequentially to prevent memory fragmentation.

3. **`get_index(int row, int col)`:** An inline stride mapping utility function calculating element positions without lookups:

$$\text{Index} = (\text{row} \times \text{shape}[1]) + \text{col}$$




## 2. The Execution Framework (`Layer` & `Activations`)

Every computational block in Valen inherits from a pure virtual interface class named `BaseForward`, enabling runtime polymorphism.

```text
                    ┌─────────────────┐
                    │   BaseForward   │  ◄── Abstract Virtual Interface
                    └────────┬────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   ┌─────────────────┐               ┌─────────────────┐
   │  Linear : Layer │               │  ReLU : Activn  │
   └─────────────────┘               └─────────────────┘

```

### Multi-Threaded Linear (Dense) Layer Optimization

The matrix multiplication engine evaluates incoming batch tensor weights through an adaptive execution scheduler gate designed to maximize throughput while minimizing operating system thread scheduling overhead:

* **Low-Throughput Gate (Batch Size $\le 100$):** Executes sequentially inside the primary caller thread to prevent thread administration overhead from degrading performance.

* **High-Throughput Gate (Batch Size $> 100$):** Fragments input batches dynamically by row spans and hands off chunks concurrently to the background execution thread pool.



## 3. Execution Topology & Deserialization (`Models`)

```text
[ Manifest Directory ] 
         │
         ▼
 ┌──────────────┐      Creates        ┌───────────────────────┐
 │ CreateModel  ├───────────────────► │    Sequential Model   │
 └──────────────┘                     │                       │
                                      │ ┌───────────────────┐ │
 [ Threadpool ] ────────────────────► │ │ std::vector of    │ │
  (Passed by Reference)               │ │ unique_ptr<Base>  │ │
                                      │ └───────────────────┘ │
                                      └───────────────────────┘

```

### Model Serialization Abstractions

* **`Sequential` Class:** Manages an execution timeline stored as a contiguous sequence of smart pointers (`std::vector<std::unique_ptr<BaseForward>>`). It exposes an internal `add()` method for structure building and a public `ForwardPass(const Nexus& input)` pipeline execution method.

* **`CreateModel` Factory:** Encapsulates design patterns to insulate the end user from build steps. Calling `create_model(dir_path, pool_ref)` dynamically parses JSON topology manifests, tracks layer configuration mappings, imports weights, and registers the global `Threadpool`.


## 4. Stateless Concurrency Layer (`Threadpool`)

Valen isolates its execution runtime threads inside a persistent, stateless tracking module to guarantee steady-state execution profiles.

```text
Work Ingestion:
  [Compute Kernel Slice] ──► Enqueue ──► [Task Queue] ──► Locked via std::mutex
                                                               │
Thread Orchestration:                                          ▼
  [Worker 1] ◄── [Worker 2] ◄── [Worker 3] ◄── Woken up via std::condition_variable
                                                               │
Thread Barrier Synchronization:                                ▼
  [Caller Pipeline Main Thread] ◄─────── Blocked via C++20 std::latch

```

1. **Persistent Resource Allocation:** Worker threads (`std::thread`) are spawned exactly once at boot time to match hardware processing limits, eliminating the kernel latency penalties of on-the-fly thread instantiation.

2. **Passive Waiting States:** Workers sit inside an operating-system managed sleep state using a condition variable (`std::condition_variable`) when the work loop queue is empty, freeing up CPU cycles for the rest of the engine.

3. **C++20 Hardware Barriers:** Threads communicate state completions back to the caller pipeline main thread via a lightweight native synchronization gate (`std::latch`). This completely avoids active memory spin-lock checking loop overhead (`std::this_thread::yield()`), reducing core processing strain by ~30%.