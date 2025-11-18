
## Requirements

```
git clone https://github.com/SurajMande/VTON.git
cd VTON

conda env create -f environment.yaml
conda activate idm
```

## Data preparation

### VITON-HD
You can download VITON-HD dataset from [VITON-HD](https://github.com/shadow2496/VITON-HD).

After download VITON-HD dataset, move vitonhd_test_tagged.json into the test folder, and move vitonhd_train_tagged.json into the train folder.

Structure of the Dataset directory should be as follows.

```

train
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_train_tagged.json

test
|-- image
|-- image-densepose
|-- agnostic-mask
|-- cloth
|-- vitonhd_test_tagged.json

```

We used [detectron2](https://github.com/facebookresearch/detectron2) for obtaining densepose images, 
```
git clone https://huggingface.co/h94/IP-Adapter
```

Move ip-adapter to ckpt/ip_adapter, and image encoder to ckpt/image_encoder.

Start training using python file with arguments,

```
accelerate launch train_xl.py \
    --gradient_checkpointing --use_8bit_adam \
    --output_dir=result --train_batch_size=6 \
    --data_dir=DATA_DIR
```

or, you can simply run with the script file.

```
sh train_xl.sh
```


## Inference


### VITON-HD

Inference using python file with arguments,

```
accelerate launch inference.py \
    --width 768 --height 1024 --num_inference_steps 30 \
    --output_dir "result" \
    --unpaired \
    --data_dir "DATA_DIR" \
    --seed 42 \
    --test_batch_size 2 \
    --guidance_scale 2.0
```

or, you can simply run with the script file.

```
sh inference.sh
```

## Start a local gradio demo 

Download checkpoints for human parsing [here](https://huggingface.co/spaces/yisol/IDM-VTON/tree/main/ckpt).

Place the checkpoints under the ckpt folder.
```
ckpt
|-- densepose
    |-- model_final_162be9.pkl
|-- humanparsing
    |-- parsing_atr.onnx
    |-- parsing_lip.onnx

|-- openpose
    |-- ckpts
        |-- body_pose_model.pth
    
```




Run the following command:

```python
python gradio_demo/app.py
```
