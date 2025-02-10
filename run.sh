# cd to your alignment-handbook path
cd ../alignment-handbook
export WANDB_API_KEY=your_wandb_api_key
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
model_list=(llama-3-8b qwen-2-7b)
dataset_list=(capybara ultrafeedback)
dpo_lambda_strategy_list=(fixed increase_linear decrease_linear)
model_choice=0 # 0: llama-3-8b, 1: qwen-2-7b
data_choice=1  # 0: capybara, 1: ultrafeedback
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
lr=5e-7
opt=paged_adamw_32bit
dpo_lambda_strategy_choice=0
dpo_lambda_strategy=${dpo_lambda_strategy_list[$dpo_lambda_strategy_choice]}
dpo_lambda_max=1.0

(
    for model_choice in 0 1
    do
        if [ $model_choice -eq 0 ]; then
            dpo_lambda_strategy_choices=("0" "1" "2")
        elif [ $model_choice -eq 1 ]; then
            dpo_lambda_strategy_choices=("0" "1" "2")
        fi
        for dpo_lambda_strategy_choice in "${dpo_lambda_strategy_choices[@]}"
        do
            if [ $dpo_lambda_strategy_choice -eq 1 ]; then
                dpo_lambda_min_values=("0.75" "0.85" "0.95")
            elif [ $dpo_lambda_strategy_choice -eq 2 ]; then
                dpo_lambda_min_values=("0.75" "0.85" "0.95")
            elif [ $dpo_lambda_strategy_choice -eq 0 ]; then
                dpo_lambda_min_values=("0.5","0.55","0.6","0.65","0.7","0.75","0.8","0.85","0.9","0.95","1.0")
            fi
            for dpo_lambda_min in "${dpo_lambda_min_values[@]}"
            do
                dpo_lambda_strategy=${dpo_lambda_strategy_list[$dpo_lambda_strategy_choice]}
                if [ "$dpo_lambda_strategy" == "fixed" ]; then
                    log_dir="../print_log/train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${lr}-${opt}-${TIMESTAMP}-${dpo_lambda_strategy}-${dpo_lambda_min}.log"
                    output_dir="/your_save_path/${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-SFTed-${opt}-${dpo_lambda_strategy}-${dpo_lambda_min}"
                    run_name="train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-${opt}-${dpo_lambda_strategy}-${dpo_lambda_min}"
                elif [ "$dpo_lambda_strategy" == "increase_linear" ]; then
                    log_dir="../print_log/train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${lr}-${opt}-${TIMESTAMP}-${dpo_lambda_strategy}-${dpo_lambda_min}to${dpo_lambda_max}.log"
                    output_dir="/your_save_path/${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-SFTed-${opt}-${dpo_lambda_strategy}-${dpo_lambda_min}to${dpo_lambda_max}"
                    run_name="train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-${opt}-${dpo_lambda_strategy}-${dpo_lambda_min}to${dpo_lambda_max}"
                elif [ "$dpo_lambda_strategy" == "decrease_linear" ]; then
                    log_dir="../print_log/train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${lr}-${opt}-${TIMESTAMP}-${dpo_lambda_strategy}-${dpo_lambda_max}to${dpo_lambda_min}.log" 
                    output_dir="/your_save_path/${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-SFTed-${opt}-${dpo_lambda_strategy}-${dpo_lambda_max}to${dpo_lambda_min}"
                    run_name="train_${model_list[$model_choice]}-dpo-${dataset_list[$data_choice]}-${TIMESTAMP}-${lr}-${opt}-${dpo_lambda_strategy}-${dpo_lambda_max}to${dpo_lambda_min}"

                fi

                # echo $log_dir
                # # echo ${model_list[$model_choice]}
                # # nohup accelerate launch \
                ACCELERATE_LOG_LEVEL=info accelerate launch \
                    --config_file your deepspeed config file \
                    --num_processes=8 scripts/run_dposhift.py "recipes/${model_list[$model_choice]}/dpo/config_full_${dataset_list[$data_choice]}.yaml" \
                    --dpo_lambda=$dpo_lambda \
                    --optim=$opt \
                    --report_to=wandb \
                    --per_device_train_batch_size=4 \
                    --gradient_accumulation_steps=4 \
                    --per_device_eval_batch_size=4 \
                    --auto_insert_empty_system_msg=False \
                    --dpo_lambda_strategy=$dpo_lambda_strategy \
                    --dpo_lambda_min=$dpo_lambda_min \
                    --dpo_lambda_max=$dpo_lambda_max \
                    --output_dir=$output_dir \
                    --run_name=$run_name \
                    > $log_dir 2>&1
                wait
            done
        done
    done
) &