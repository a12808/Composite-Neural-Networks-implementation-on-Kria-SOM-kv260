
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iomanip> // necessário para std::fixed e std::setprecision

#include <glob.h>
#include <opencv2/opencv.hpp>

#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

#include <vart/tensor_buffer.hpp>
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <vart/runner_helper.hpp>

#include <vart/assistant/xrt_bo_tensor_buffer.hpp>


using namespace std;
using namespace std::chrono;
using namespace vart;


// -----------------------------
// Paths
// -----------------------------
#define DATASET_DIR                     "/home/root/imagenet500/images/"
#define LABELS_PATH                     "/home/root/imagenet500/words.txt"
#define VAL_LABELS_PATH                 "/home/root/imagenet500/val.txt"

#define XMODEL_PATH                     "test_2_compiled.xmodel"

#define TEST_B3_RESULT_CSV_PATH         "test_B3_kv260_results_cpp.csv"
#define TEST_B3_POWER_RESULT_CSV_PATH   "test_B3_kv260_power_results_cpp.csv"

// power thread
#define SYSFS_POWER_PATH                "/sys/class/hwmon/hwmon0/power1_input"
#define POWER_LOG_INTERVAL_S            0.01
atomic<bool> g_powerlog_stop(false);
thread g_powerlog_thread;


// -------------------------
// Simple DEBUG Function
// -------------------------
template<typename... Args>
void DEBUG(Args... args)
{
    std::cout << "[DBG] ";
    // Expande os argumentos no cout
    ((std::cout << args), ...) << std::endl;
}
void DEBUG(const string& message) 
{
    std::cout << "[DBG] " << message << std::endl;
}
double get_timestamp() 
{
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double>(duration).count();
}

// -------------------------
// Load labels
// -------------------------
vector<string> Load_Labels(const string& path)
{
    vector<string> labels;
    ifstream f(path);
    string line;

    while(getline(f, line))
    {
        labels.push_back(line);
    }
    
    return labels;
}

// -------------------------
// Load validation labels mapping
// -------------------------
unordered_map<string,int> Load_ValLabels(const string& path)
{
    unordered_map<string,int> mapping;
    ifstream f(path);
    string line;

    while(getline(f, line))
    {
        auto sep = line.find(' ');
        string name = line.substr(0, sep);
        int label = stoi(line.substr(sep+1));
        mapping[name] = label;
    }
    
    return mapping;
}

// -------------------------
// Load Dataset
// -------------------------
vector<string> Load_Dataset(const string& dataset_dir)
{
    vector<string> image_files;
    glob_t glob_result;
    string pattern = dataset_dir + "/*.JPEG";

    int ret = glob(pattern.c_str(), 0, NULL, &glob_result);

    if(ret != 0)
    {
        DEBUG("No images found in", dataset_dir);
        return image_files;
    }

    for(size_t i=0; i<glob_result.gl_pathc; ++i)
    {
        image_files.push_back(glob_result.gl_pathv[i]);
    }
    
    globfree(&glob_result);

    return image_files;
}

// -------------------------
// Preprocess (resize + center crop + RGB)
// -------------------------
cv::Mat Preprocess(const cv::Mat& img, int resize_short=256, int crop_size=224)
{
    int h = img.rows;
    int w = img.cols;
    int new_h, new_w;

    if(h < w)
    {
        new_h = resize_short;
        new_w = w * resize_short / h;
    } 
    else 
    {
        new_w = resize_short;
        new_h = h * resize_short / w;
    }

    cv::Mat r;
    cv::resize(img, r, cv::Size(new_w, new_h));

    int y0 = (new_h - crop_size) / 2;
    int x0 = (new_w - crop_size) / 2;
    cv::Rect roi(x0, y0, crop_size, crop_size);
    cv::Mat crop = r(roi);

    cv::cvtColor(crop, crop, cv::COLOR_BGR2RGB);
    crop.convertTo(crop, CV_8S); // xint8 / int8

    return crop;
}

// -------------------------
// Postprocess top-k
// -------------------------
pair<int, float> postprocess_top1(const vector<float>& logits) 
{
    // Implement your postprocessing logic here
    // Find the index with maximum value and return it with the probability

    if (logits.empty()) return {-1, 0.0f};

    // Encontrar o valor máximo para estabilidade numérica
    float max_val = *std::max_element(logits.begin(), logits.end());

    // Calcular exponenciais
    std::vector<float> exps(logits.size());
    float sum_exps = 0.0f;
    
    for (size_t i = 0; i < logits.size(); i++) 
    {
        exps[i] = std::exp(logits[i] - max_val); // subtrair max para estabilidade
        sum_exps += exps[i];
    }
    
    // Calcular probabilidades softmax
    std::vector<float> probs(logits.size());
    for (size_t i = 0; i < logits.size(); i++) 
    {
        probs[i] = exps[i] / sum_exps;
    }
    
    // Encontrar o índice com maior probabilidade
    int max_index = 0;
    float max_prob = probs[0];
    
    for (size_t i = 1; i < probs.size(); i++) {

        if (probs[i] > max_prob) 
        {
            max_prob = probs[i];
            max_index = i;
        }
    }
    
    return {max_index, max_prob};
}

// -----------------------------
// Process images
// -----------------------------
void debug_subgraphs(const std::vector<xir::Subgraph*>& subgraphs) 
{
    std::cout << "=== DPU SUBGRAPHS DEBUG INFO ===" << std::endl;
    std::cout << "Found " << subgraphs.size() << " DPU subgraph(s)\n" << std::endl;
    
    for (size_t i = 0; i < subgraphs.size(); ++i) 
    {
        auto* sg = subgraphs[i];  // raw pointer, não owning
        std::cout << "[" << i << "] " << sg->get_name() << std::endl;
        
        // runner espera Subgraph* — já temos
        auto runner = vart::Runner::create_runner(sg, "run");
        auto inputs = runner->get_input_tensors();
        auto outputs = runner->get_output_tensors();
        
        // Print input shapes
        std::cout << "    Inputs:  ";
        for (auto& tensor : inputs) 
        {
            auto dims = tensor->get_shape();
            std::cout << "[";
            for (size_t j = 0; j < dims.size(); ++j) 
            {
                std::cout << dims[j];
                if (j < dims.size() - 1) std::cout << ", ";
            }
            std::cout << "] ";
        }
        std::cout << std::endl;
        
        // Print output shapes
        std::cout << "    Outputs: ";
        for (auto& tensor : outputs) 
        {
            auto dims = tensor->get_shape();
            std::cout << "[";
            for (size_t j = 0; j < dims.size(); ++j) 
            {
                std::cout << dims[j];
                if (j < dims.size() - 1) std::cout << ", ";
            }
            std::cout << "] ";
        }
        std::cout << "\n" << std::endl;
    }
    std::cout << "=================================" << std::endl;
}

/**
 * Two "paralel" inference with ReLu
 */
void TestB3(const std::vector<string>& image_paths, 
            const std::vector<string>& labels, 
            const std::unordered_map<std::string, int>& val_mapping, 
            const std::string& results_csv)
{
    std::ofstream file(results_csv);

    if (!file.is_open()) 
    {
        throw std::runtime_error("Cannot open CSV file: " + results_csv);
    }

    file << std::fixed << std::setprecision(6); // precisão de 6 casas decimais
    file << "img_name,"
         << "pred_lbl_1,pred_name_1,prob_1,"
         << "pred_lbl_2,pred_name_2,prob_2,"
         << "gt_lbl,gt_name,"
         << "load_time_1,load_time_2,preprocess_time_1,preprocess_time_2,"
         << "upload_time,inference_time,"
         << "download_time,postprocess_time,"
         << "total_time\n";

    // Load compiled graph and find DPU subgraph
    auto graph = xir::Graph::deserialize(XMODEL_PATH);
    auto root = graph->get_root_subgraph();
    auto child_subgraphs = root->children_topological_sort();
    
    // non-owning pointers !!!
    std::vector<xir::Subgraph*> subgraphs;

    for (auto& sg : child_subgraphs) 
    {
        if (sg->has_attr("device")) 
        {
            auto device = sg->get_attr<std::string>("device");
            transform(device.begin(), device.end(), device.begin(), ::toupper);
            if (device == "DPU") subgraphs.push_back(sg); // store raw pointer, do NOT own it
        }
    }
    
    debug_subgraphs(subgraphs);

    if (subgraphs.empty()) 
    {
        throw std::runtime_error("No DPU subgraph found in " XMODEL_PATH);
    }

    auto runner = vart::Runner::create_runner(subgraphs[0], "run");

    // Dataset loop
    for (size_t i = 0; i < image_paths.size(); ++i) 
    {
        DEBUG(i);
    
        // gt
        string img_path = image_paths[i];
        string img_name = img_path.substr(img_path.find_last_of("/\\") + 1);
        int gt_lbl = -1;
        auto it = val_mapping.find(img_name);
        if (it != val_mapping.end()) gt_lbl = it->second;
        std::string gt_name = (gt_lbl >= 0 && gt_lbl < labels.size()) ? labels[gt_lbl] : "UNK";

        auto total_time_start = get_timestamp();

        // -------------------------------
        // IMAGES LOAD
        // -------------------------------
        auto t_start = get_timestamp();

        cv::Mat img_1 = cv::imread(img_path);

        if (img_1.empty()) 
        {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }
        
        double load_time_1 = get_timestamp()- t_start;

        t_start = get_timestamp();

        cv::Mat img_2 = cv::imread(img_path);

        if (img_2.empty()) 
        {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }
        
        double load_time_2 = get_timestamp()- t_start;


        // -------------------------------
        // PREPROCESS
        // -------------------------------
        t_start = get_timestamp();

        cv::Mat blob_1 = Preprocess(img_1);

        double preprocess_time_1 = get_timestamp() - t_start;

        t_start = get_timestamp();

        cv::Mat blob_2 = Preprocess(img_2);

        double preprocess_time_2 = get_timestamp() - t_start;


        // -------------------------------
        // DPU BUFFERS and UPLOAD
        // -------------------------------

        t_start = get_timestamp();

        // como as imagens agora vão juntas, assumo que alocação de buffer é upload
        // Obter os descritores de input/output
        auto input_tensors  = runner->get_input_tensors();
        auto output_tensors = runner->get_output_tensors();

        // Criar buffers para inputs/outputs
        std::vector<std::unique_ptr<vart::TensorBuffer>> input_tensor_buffers_owner;
        std::vector<std::unique_ptr<vart::TensorBuffer>> output_tensor_buffers_owner;

        std::vector<vart::TensorBuffer*> input_tensor_buffers;
        std::vector<vart::TensorBuffer*> output_tensor_buffers;

        // Alocar buffers de input
        for (auto& tensor : input_tensors) 
        {
            auto buffer = vart::alloc_cpu_flat_tensor_buffer(tensor);
            input_tensor_buffers_owner.push_back(std::move(buffer));
            input_tensor_buffers.push_back(input_tensor_buffers_owner.back().get());
        }

        // Alocar buffers de output  
        for (auto& tensor : output_tensors) 
        {
            auto buffer = vart::alloc_cpu_flat_tensor_buffer(tensor);
            output_tensor_buffers_owner.push_back(std::move(buffer));
            output_tensor_buffers.push_back(output_tensor_buffers_owner.back().get());
        }

        // Para cada buffer de input, copiar blobs para os buffers
        for (size_t i = 0; i < input_tensor_buffers.size(); ++i)
        {
            auto* input_buffer = input_tensor_buffers[i];

            // Fazer sync para escrita
            input_buffer->sync_for_write(0, input_buffer->get_tensor()->get_data_size());
            
            // Obter ponteiro para os dados
            auto data_ptr = input_buffer->data().first;
            size_t buffer_size = input_buffer->get_tensor()->get_data_size();
            
            // Selecionar blob_1 ou blob_2 conforme o input
            cv::Mat& blob = (i == 0) ? blob_1 : blob_2;

            // Copiar dados das imagens processada
            std::memcpy((void*)data_ptr, blob.data, std::min(buffer_size, (size_t)(blob.total() * blob.elemSize())));
        }

        // sync input tensor buffers
        for (auto& input : input_tensor_buffers) 
        {
            input->sync_for_write(0, input->get_tensor()->get_data_size() / input->get_tensor()->get_shape()[0]);
        }

        double upload_time = get_timestamp() - t_start;


        // -------------------------------
        // INFERENCE
        // -------------------------------
        t_start = get_timestamp();

        auto run_inference = [](vart::Runner* runner,
                                const std::vector<vart::TensorBuffer*>& input_buffers,
                                const std::vector<vart::TensorBuffer*>& output_buffers) 
        {
            auto job_id = runner->execute_async(input_buffers, output_buffers);
            runner->wait(job_id.first, -1);
        };

        std::thread t(run_inference, runner.get(), input_tensor_buffers, output_tensor_buffers);
        t.join();

        double inference_time = get_timestamp() - t_start;


        // -------------------------------
        // DOWNLOAD
        // -------------------------------

        t_start = get_timestamp();

        // Fazer sync dos outputs para leitura
        for (auto& output_buffer : output_tensor_buffers) 
        {
            output_buffer->sync_for_read(0, output_buffer->get_tensor()->get_data_size());
        }

        double download_time = get_timestamp() - t_start;


        // -------------------------------
        // POST PROCESS
        // -------------------------------
        t_start = get_timestamp();
                
        // 1
        int pred_lbl_1           = -1;
        int pred_lbl_2           = -1;
        float prob_1             = 0.0f;
        float prob_2             = 0.0f;
        std::string label_name_1 = "UNK";
        std::string label_name_2 = "UNK";
        
        auto tensor = output_tensor_buffers[0]->get_tensor();
        int size = tensor->get_element_num();
        std::vector<float> logits(size);

        try 
        {
            int fp = std::any_cast<int>(tensor->get_attr("fix_point"));
            float scale = std::pow(2.0f, -fp);
            
            auto data_ptr = reinterpret_cast<int8_t*>(output_tensor_buffers[0]->data().first);
            for (int j = 0; j < size; j++) 
            {
                logits[j] = static_cast<float>(data_ptr[j]) * scale;
            }
        } 
        catch (...) 
        {
            auto data_ptr = reinterpret_cast<int8_t*>(output_tensor_buffers[0]->data().first);
            for (int j = 0; j < size; j++) 
            {
                logits[j] = static_cast<float>(data_ptr[j]);
            }
        }
        
        // split metade/metade
        int half = size / 2;
        std::vector<float> logits1(logits.begin(), logits.begin() + half);
        std::vector<float> logits2(logits.begin() + half, logits.end());

        std::tie(pred_lbl_1, prob_1) = postprocess_top1(logits1);
        std::tie(pred_lbl_2, prob_2) = postprocess_top1(logits2);

        if (pred_lbl_1 >= 0 && pred_lbl_1 < (int)labels.size()) label_name_1 = labels[pred_lbl_1];
        if (pred_lbl_2 >= 0 && pred_lbl_2 < (int)labels.size()) label_name_2 = labels[pred_lbl_2];

        double postprocess_time = get_timestamp() - t_start;

        double total_time = get_timestamp() - total_time_start;

        // -------------------------------
        // CSV
        // -------------------------------

        /*
        file << "img_name,"
         << "pred_lbl_1,pred_name_1,prob_1,"
         << "pred_lbl_2,pred_name_2,prob_2,"
         << "gt_lbl,gt_name,"
         << "load_time_1,load_time_2,preprocess_time_1,preprocess_time_2,"
         << "upload_time,inference_time,"
         << "download_time,postprocess_time,"
         << "total_time\n";
        */

        file << img_name        << ","
             << pred_lbl_1      << "," << "\"" << label_name_1 <<  "\"" << "," << prob_1 << ","
             << pred_lbl_2      << "," << "\"" << label_name_2 <<  "\"" << "," << prob_2 << ","
             << gt_lbl          << "," << "\"" << gt_name << "\""      << ","
             << load_time_1     << "," << load_time_2 << "," << preprocess_time_1 << "," << preprocess_time_2 << ","
             << upload_time     << "," << inference_time               << ","
             << download_time   << "," << postprocess_time             << "," 
             << total_time << "\n";
    }
    
    graph.reset();
    runner.reset(); 
    file.close();
}
 

// -------------------------
// Power logger 
// -------------------------
void kv260_power_logger_sysfs(const string& csv_path, atomic<bool>& stop_event, float interval, const string& power_path)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto next_time = start;

    std::ofstream file(csv_path);

    if (!file.is_open()) 
    {
        cout << "[ERROR] Cannot open power log file: " << csv_path << endl;
        return;
    }

    DEBUG("KV260 power logging started " + csv_path);
    file << "timestamp,power_watts\n";

    try
    {
        while (!stop_event.load()) 
        {
            auto now = chrono::duration<double>(chrono::high_resolution_clock::now() - start).count();

            try 
            {
                ifstream power_file(power_path);
            
                if (power_file.is_open()) 
                {
                    string line;
                    getline(power_file, line);
                    int uw = stoi(line);
                    file << now << "," << (uw / 1e6) << "\n";
                    file.flush();
                }
            } 
            catch (const exception& e) 
            {
                cout << "[WARN] power read failed: " << e.what() << endl;
            }

            // Alinhamento de tempo
            next_time += chrono::milliseconds(static_cast<int>(interval * 1000));
            auto sleep_time = chrono::duration<double>(next_time - chrono::high_resolution_clock::now());
            
            if (sleep_time.count() > 0) 
            {
                this_thread::sleep_for(sleep_time);
            } 
            else 
            {
                next_time = chrono::high_resolution_clock::now();
            }
        }
    }
    catch (const exception& e) 
    {
        cout << "[ERROR] Power logger exception: " << e.what() << endl;
    }

    file.close();
}

void Start_kv260_power_log(const std::string& csv_path) 
{
    g_powerlog_stop.store(false);

    g_powerlog_thread = thread(
        kv260_power_logger_sysfs,
        csv_path,
        std::ref(g_powerlog_stop),
        POWER_LOG_INTERVAL_S,
        SYSFS_POWER_PATH
    );
}

void Stop_kv260_power_log(void) 
{
    g_powerlog_stop.store(true);
    
    // Dar tempo para o thread terminar gracefuly
    if (g_powerlog_thread.joinable()) 
    {
        // Esperar com timeout para evitar deadlock
        auto timeout = std::chrono::seconds(2);
        if (g_powerlog_thread.joinable()) 
        {
            g_powerlog_thread.join();
        }
    }

    DEBUG("KV260 power logging stopped");
}


// -------------------------
// Main
// -------------------------
int main(int argc, char** argv)
{
    // Start power logger
    Start_kv260_power_log(TEST_B3_POWER_RESULT_CSV_PATH);

    // Load Labels
    auto labels = Load_Labels(LABELS_PATH);
    auto val_mapping = Load_ValLabels(VAL_LABELS_PATH);
    DEBUG("Labels loaded");

    // Load Dataset
    auto image_files = Load_Dataset(DATASET_DIR);
    DEBUG("Found ", image_files.size(), " images in ", DATASET_DIR);

    // Process images
    DEBUG("Processing ... ");
    TestB3(image_files, labels, val_mapping, TEST_B3_RESULT_CSV_PATH);

    // Stop power logger
    Stop_kv260_power_log();

    return 0;
}
