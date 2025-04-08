#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <openvino/openvino.hpp>

class Embeddings {
public:
    ov::Core core;
    ov::InferRequest embedding_model;
    ov::InferRequest tokenizer;
    size_t BATCH_SIZE = 1;

    // Initialize the models and tokenizer
    void init(const std::string& encoder_path,
            const std::string& tokenizer_path,
            const std::string& device,
            const std::string& dll_path,
            const std::string& ov_cache_dir = "./ov_cache") 
    {
        if (!std::filesystem::exists(dll_path)) {
            std::cerr << "[ERROR] Tokenizer DLL not found: " << dll_path << "\n";
            return;
        }

        try {
            core.add_extension(dll_path);
            std::cout << "Loaded tokenizer DLL.\n";

            if (!ov_cache_dir.empty()) {
                std::filesystem::create_directories(ov_cache_dir); 
                core.set_property(ov::cache_dir(ov_cache_dir));
                std::cout << "Set OpenVINO cache directory to: " << ov_cache_dir << std::endl;
            }

            embedding_model = core.compile_model(encoder_path, device).create_infer_request();
            tokenizer = core.compile_model(tokenizer_path, "CPU").create_infer_request();

            std::cout << "Models initialized successfully.\n";
        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Initialization failed: " << e.what() << std::endl;
        }
    }


    // Encode multiple queries in a batch
    std::vector<std::vector<float>> encode_queries(const std::vector<std::string>& queries) {
        std::vector<std::vector<float>> embedding_results;
        for (const auto& q : queries) {
            embedding_results.push_back(encode_query(q));
        }
        return embedding_results;
    }

private:
    // Encode a single query
    std::vector<float> encode_query(const std::string& query) {
        std::cout << "  encode_query: Starting for query: \"" << query << "\"\n";

        try {
            std::vector<ov::Tensor> tokenized;
            bool with_token_type_ids;
            // Tokenize the input query
            std::tie(tokenized, with_token_type_ids) = tokenize(query);

            // Pad tensors to match the model's expected input shape
            auto input_ids = pad_tensor(tokenized[0], {1, embedding_model.get_input_tensor(0).get_shape()[1]});
            auto attention_mask = pad_tensor(tokenized[1], {1, embedding_model.get_input_tensor(0).get_shape()[1]});

            // Set input tensors for the embedding model
            embedding_model.set_tensor("input_ids", input_ids);
            embedding_model.set_tensor("attention_mask", attention_mask);
            
            // Only set token_type_ids if they are available
            if (with_token_type_ids) {
                auto token_type_ids = pad_tensor(tokenized[2], {1, embedding_model.get_input_tensor(0).get_shape()[1]});
                embedding_model.set_tensor("token_type_ids", token_type_ids);
            }

            // Perform inference
            embedding_model.infer();
            // std::cout << "  encode_query: embedding_model.infer() successful.\n";

            // Retrieve the output tensor
            auto result_tensor = embedding_model.get_output_tensor(0);
            auto shape = result_tensor.get_shape();
            std::cout << "  embedding_model output shape " << shape << std::endl;
            float* data = result_tensor.data<float>();

            // Convert the output tensor to a vector
            std::vector<float> result(data, data + shape[1]);
            return result;
        } catch (const std::exception& ex) {
            std::cerr << "[ERROR] Encoding failed: " << ex.what() << "\n";
            return {};
        }
    }

    std::pair<std::vector<ov::Tensor>, bool> tokenize(std::string prompt) {
        auto input_tensor = ov::Tensor{ov::element::string, {BATCH_SIZE}, &prompt};
        bool with_token_type_ids = true;
    
        try {
            tokenizer.set_input_tensor(input_tensor);
        } catch (const std::exception& e) {
            std::cerr << e.what() << '\n';
            return {{}, false};  // If the tokenizer setup fails, return empty vector and false
        }
    
        tokenizer.infer();
    
        
        ov::Tensor input_ids(tokenizer.get_tensor("input_ids").get_element_type(), tokenizer.get_tensor("input_ids").get_shape());
        tokenizer.get_tensor("input_ids").copy_to(input_ids);
        
        ov::Tensor attention_mask(tokenizer.get_tensor("attention_mask").get_element_type(), tokenizer.get_tensor("attention_mask").get_shape());
        tokenizer.get_tensor("attention_mask").copy_to(attention_mask);
        
        // Check if token_type_ids tensor is available
        ov::Tensor token_type_ids;
        try {
            token_type_ids = tokenizer.get_tensor("token_type_ids");
            
            with_token_type_ids = true;  // Set flag if tensor exists
        } catch (...) {
            with_token_type_ids = false;
        }
        
        if (with_token_type_ids) {
            ov::Tensor token_type_ids(tokenizer.get_tensor("token_type_ids").get_element_type(), tokenizer.get_tensor("token_type_ids").get_shape());
            tokenizer.get_tensor("token_type_ids").copy_to(token_type_ids);
        }
    
        // Return both tensors and the success flag (with_token_type_ids)
        return {{input_ids, attention_mask, token_type_ids}, with_token_type_ids};
    }
    

    // Pad a tensor to the target shape with zeros
    ov::Tensor pad_tensor(const ov::Tensor& input, const ov::Shape& target_shape) {
        ov::Tensor padded{ov::element::i64, target_shape};
        std::fill_n(padded.data<int64_t>(), padded.get_size(), 0);
        std::copy_n(input.data<int64_t>(), input.get_size(), padded.data<int64_t>());
        return padded;
    }
};


void Benchmark(ov::InferRequest embedding_model, int niter) {
    // Check if the number of iterations is greater than zero
    if (niter > 0) {
        std::vector<int64_t> latencies;
        latencies.reserve(niter);
        auto start = std::chrono::steady_clock::now();
        auto time_point = start;

        // Run the inference for 'niter' times
        for (int i = 0; i < niter; ++i) {
            embedding_model.infer();

            auto iter_end = std::chrono::steady_clock::now();
            auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - time_point).count();
            latencies.push_back(time_ms);
            time_point = iter_end;
        }

        // Sort the latencies to compute min, max, and average
        std::sort(latencies.begin(), latencies.end());
        auto min = latencies[0];
        auto avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        auto max = latencies.back();

        // Output the latency results
        std::cout << "Latency:" << std::endl;
        std::cout << "   Average: " << avg << " ms" << std::endl;
        std::cout << "   Min:     " << min << " ms" << std::endl;
        std::cout << "   Max:     " << max << " ms" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string encoder_path = "./models/bert/openvino_model.xml";
    std::string tokenizer_path = "./models/bert/openvino_tokenizer.xml";
    std::string device = "CPU";
    int niter = 1;
    std::string ov_cache_dir = "./ov_cache";

    std::cout << ov::get_openvino_version() << std::endl;

    if (argc == 5) {
        encoder_path = argv[1];
        tokenizer_path = argv[2];
        device = argv[3];

        std::istringstream ss(argv[4]);
        if (!(ss >> niter)) {
            std::cerr << "[ERROR] Invalid number for num_of_iterations: " << argv[4] << '\n';
            return EXIT_FAILURE;
        }
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0] << " <encoder_model.xml> <tokenizer_model.xml> <device_name> <num_of_iterations>\n";
        return EXIT_FAILURE;
    }

    // Get the path to the tokenizer DLL
    std::filesystem::path exe_dir = std::filesystem::path(argv[0]).parent_path();
    std::string dll_path = (exe_dir / "openvino_tokenizers.dll").string();

    // Initialize the embedder
    Embeddings embedder;
    embedder.init(encoder_path, tokenizer_path, device, dll_path, ov_cache_dir);

    // Define sample queries(copy test chunks from document indexing: loader and splitter)
    std::vector<std::string> queries = {
        "Hello, world!",
        "This is an embedding test.",
        // "OpenVINO tokenizer + BERT in C++."
    };

    auto embeddings = embedder.encode_queries(queries);

    // acc checking: print the first 5 values of each embedding
    for (size_t j = 0; j < embeddings.size(); ++j) {
        std::cout << "Embedding for \"" << queries[j] << "\":(first 5 values)\n";
        int count = 0;
        for (float v : embeddings[j]) {
            if (count >= 5) break;
            std::cout << v << " ";
            count++;
        }
        std::cout << "\n";
    }

    // Latency benchmarking after warmup
    Benchmark(embedder.embedding_model, niter);

    return 0;
}