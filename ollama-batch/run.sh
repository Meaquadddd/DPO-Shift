
dataset_names=('capybara' 'ultrafeedback' 'alpacafarm')
model_names=('meta-llama' 'mistral' 'Qwen')
real_names=('Llama-3-8b' 'Zephyr' 'Qwen-7b')
strategy=('fixed' 'increase_linear' 'decrease_linear')
test_dataset_choice=1
train_dataset_choice=1
model_choice=2
strtegy_choice=0
dpo_lambda=0.85
sft_baseline=0

# prompt=Qwen-7b-dpo-ultrafeedback-decrease_linear-1.0to0.95-VSDPO-capybara_new.jsonl
# prompt=Qwen-7b-dpo-ultrafeedback-fixed-0.95-VSDPO-capybara_new.jsonl

(
    for test_dataset_choice in 0 1
    do
        for model_choice in 0
        do
            for strategy_choice in 0
            do
                if [ $strategy_choice -eq 0 ]; then
                    dpo_lambda_min_values=("0.55" "0.65" "0.75" "0.85" "0.95" "sft")
                    # dpo_lambda_min_values=("0.96" "0.97" "0.98" "0.99")
                    # dpo_lambda_min_values=("0.95" "1.0")
                    # dpo_lambda_min_values=("0.999" "0.9999")
                elif [ $strategy_choice -eq 2 ]; then
                    dpo_lambda_min_values=("0.95")
                elif [ $strategy_choice -eq 1 ]; then
                    dpo_lambda_min_values=("0.95")
                fi
                TIMESTAMP=$(date +"%Y%m%d-%H%M%S")

                for dpo_lambda in "${dpo_lambda_min_values[@]}"
                do
                    if [ "$dpo_lambda" == "sft" ]; then
                        prompt="./prompts/${real_names[$model_choice]}-sft-ultrachat_200k-${dataset_names[$test_dataset_choice]}.jsonl"
                        judge_path="judges/${real_names[$model_choice]}-sft-ultrachat_200k-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                        answer_path="answers/${real_names[$model_choice]}-sft-ultrachat_200k-${dataset_names[$test_dataset_choice]}-answer"
                    else
                        if [ $strategy_choice -eq 0 ]; then
                            prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft.jsonl"
                            fi
                            judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-judge-new.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft-judge-new.jsonl"
                            fi
                            answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-answer-new"
                            if [ $sft_baseline -eq 1 ]; then
                                answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft-answer-new"
                            fi
                        elif [ $strategy_choice -eq 1 ]; then
                            prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-vssft.jsonl"
                            fi
                            judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-vssft-judge.jsonl"
                            fi
                            answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-answer"
                            if [ $sft_baseline -eq 1 ]; then
                                answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-vssft-answer"
                            fi
                        elif [ $strategy_choice -eq 2 ]; then
                            prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0to${dpo_lambda}-${dataset_names[$test_dataset_choice]}.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0to${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft.jsonl"
                            fi
                            judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                            if [ $sft_baseline -eq 1 ]; then
                                judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft-judge.jsonl"
                            fi
                            answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-answer"
                            if [ $sft_baseline -eq 1 ]; then
                                answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-vssft-answer"
                            fi
                        fi
                        
                        
                    fi

                    echo "Start processing $prompt"
                    echo "Output to $judge_path"
                    echo "Output to $answer_path" 

            
                    python ./ollama-batch-process.py \
                        --prompts $prompt \
                        --output $answer_path \
                        > print_log.log 2>&1

                    python ./response-json-merge.py \
                        --input-dir $answer_path \
                        --output-file $judge_path \

                    wait
                done
            done
        done
    done
) &