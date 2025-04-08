# ov_text_embedding_sample
OpenVINO native API pipeline for text embedding.
Currently support cpp and python for BGE models.

### Model conversion:
```bash
pip install -r requirements.txt
python convert_quantize-ov_embedding-bge-m3-gpu.py
```
You can also convert bge-small-zh-v1.5 with script `convert_quantize_ov_embedding.py` or modify this script for other BGE models.
### Acc checking:
Compare generation with pytorch model via Mean Squared Error (MSE) 
```bash
python compare-ov_embedding-bge-m3-int8-int4.py
```

### Setup for C++ deployment:
Use Windows Command Prompt:
```bash
curl -O https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/pre-release/2025.1.0.0rc3/openvino_genai_windows_2025.1.0.0rc3_x86_64.zip
tar -xf openvino_genai_windows_2025.1.0.0rc3_x86_64.zip
openvino_genai_windows_2025.1.0.0rc3_x86_64\setupvars.bat
cmake -S .\ -B .\build\ && cmake --build .\build\ --config Release -j8
xcopy "openvino_genai_windows_2025.1.0.0rc3_x86_64\runtime\bin\intel64\Release\*.dll" ".\build\Release" /s /i
xcopy "openvino_genai_windows_2025.1.0.0rc3_x86_64\runtime\3rdparty\tbb\bin\*.dll" ".\build\Release" /s /i
```

### Usage for GPU benchmarking:
```bash
build\Release\bge_sample.exe bge-m3-ov\openvino_model_int8_bs_1.xml bge-m3-ov\openvino_tokenizer.xml GPU 3
build\Release\bge_sample.exe bge-small-zh-v1.5\openvino_model_bs_1.xml bge-small-zh-v1.5\openvino_tokenizer.xml NPU 3
```

### Notice:
bge-m3 is large one, so we convert fp16 model into int8 model via NNCF PTQ and do inference on GPU for best performance.
bge-m3 has only 2 inputs, while other models from BGE family have 3 inputs.
Here C++ pipeline could be used for all models from BGE family without modification.
For NPU inference, please use scipt to convert the model with static shape. 