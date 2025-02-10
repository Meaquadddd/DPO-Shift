# Training

1. our code is developed based on the `alignment-handbook`, please install it following `https://github.com/huggingface/alignment-handbook`
2. add `args.max_steps = max_steps` in the `trainer.py` from `transformer`, this is necessary for the `linear_increase` and `linear_decrease` strategies. This is just a rude solution to track the total numbers of optimization steps, you are welcome to modify it.
3. run `bash ./run.sh` to start training

# Evaluation

1. Run MT Bench and generate answer from Model 
 ```Shell
cd Fastchat/fastchat/llm_judge
bench_list=(mt_bench ultrafeedback capybara alpacafarm)

folder_path = YOUR_MODEL_PATH

model_id = YOUR_MODEL_ID

python ./gen_model_answer.py \
        --model-path $folder_path \
        --model-id $model_id \
        --bench-name ${bench_list[$bench_choice]} \
        --num-gpus-total 8 \
        > gen_answer_print_log.log 2>&1

```
2. Or you can list all the model you need to evaluate in the `latest_po_info.txt` (do follow the format) and run
 ```Shell
bash ./mtbench.sh
``` 
4. test win rate with ollama
set `OLLAMA_BINARY` in `ollama-batch-servers` with your ollama binary path
 ```Shell
 cd ollama-batch
 bash ./ollama-batch-servers numbers_of_your_GPU # start ollama processes
 bash ./run.sh # to generate judges
```
4. the judgment is recorded in the `judges` files
