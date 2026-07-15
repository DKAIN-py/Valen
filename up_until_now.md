1. This project is a inference engine for the NexusDL.
2. Currently valen has capability to run MLPs made from NexusDL.

## How does this work?

### Nexus
-- Primitive tensor data structure is the same as Nexus in the NexusDL.
-- Nexus object has atrribute:
    - tensor shape: a vector of integers, first element represent the batchsize, rest represent the shape of matrix in order.
    - data : a 1D vector of float(32-bit) which stores data from a tensor into this form:
        -- data = [ {Row 1}, {Row 2}, .... , {Row n} ]
        -- Here Row represent stragiht numbers.
-- Nexus has function to load data from various methods:
    - load_from_binary : used to create model from the model manifest, load data from the .bin file directly (.bin file should also be a flat binary file)
    - load_input : can be used to convert any predefiend tensor into the flat data vector.
-- Nexus has get_index(int row, int col) method to retriev a element from the data vector from the POV of a matrix(to be made into a general tensor and be replaced by pointers in the forward pass).

### Layer
-- Currently there is only Linear layer.
    - Linear layer has its own weight and bias Nexus object that are initialised during the model creation.
    - Forward is multithread optimized for batchsize > 100, for batchsize<=100 single (main) thread does the execution.

### Activations
-- Currently there is only ReLU in the valen.

### Models
-- Currently we have Sequential model.
    - It is vector of BaseForward class objects, which every Layer and Activation inherits so that we can loop over the model through BaseForward class pointer.
    - Sequential model lets us run Sequential models in the correct manner.
    - Sequential model has a add(std::unique_ptr<BaseForward> layer) to add layers to the model, it is used internaly only.
    - It also has ForwardPass(const Nexus& input) method to do the inference.

### Model Creation
-- We bind Sequential class into a CreateModel class to abstract away the creation from the user.
-- CreateModel has only one method called create_model(const std::string& dir_path, const Threadpool& pool_ref)
    - dir_path : it the path of the directory where model's manifest and weight files are stored.
    - pool_ref : it the refernce of a global Threadpool object to work as a stateless independent mutlithreading way.

### Threadpool
-- Made a generic Threadpool class to work as a stateless entity for mutlithreading whereever necceasarry.
-- Currently used in parrallelzing the matrix multiplication in the Linear layer.