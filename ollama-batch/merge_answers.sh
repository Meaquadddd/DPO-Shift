
dataset_names=('capybara' 'ultrafeedback' 'alpacafarm')
model_names=('meta-llama' 'mistral' 'Qwen')
real_names=('Llama-3-8b' 'Zephyr' 'Qwen-7b')
strategy=('fixed' 'increase_linear' 'decrease_linear')
test_dataset_choice=1
train_dataset_choice=1
model_choice=2
strtegy_choice=0
dpo_lambda=0.85
use_old=0

# prompt=Qwen-7b-dpo-ultrafeedback-decrease_linear-1.0to0.95-VSDPO-capybara_new.jsonl
# prompt=Qwen-7b-dpo-ultrafeedback-fixed-0.95-VSDPO-capybara_new.jsonl

(
    for model_choice in 2
    do
        for strategy_choice in 0
        do
            if [ $strategy_choice -eq 0 ]; then
                # dpo_lambda_min_values=("0.55" "0.65" "0.75" "0.85" "0.95" "sft")
                dpo_lambda_min_values=("0.95")
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
                        if [ $use_old -eq 1 ]; then
                            prompt="./prompts_new/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}.jsonl"
                            judge_path="judges_/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                            answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-answer"
                        else
                            prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}.jsonl"
                            judge_path="judges_/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-judge-new.jsonl"
                            answer_path="answers_/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}-${dataset_names[$test_dataset_choice]}-answer-new"
                        fi
                    elif [ $strategy_choice -eq 1 ]; then
                        prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}.jsonl"
                        judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                        answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-${dpo_lambda}to1.0-${dataset_names[$test_dataset_choice]}-answer"
                    elif [ $strategy_choice -eq 2 ]; then
                        prompt="./prompts/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0to${dpo_lambda}-${dataset_names[$test_dataset_choice]}.jsonl"
                        judge_path="judges/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-judge.jsonl"
                        answer_path="answers/${real_names[$model_choice]}-dpo-${dataset_names[$train_dataset_choice]}-${strategy[$strategy_choice]}-1.0${dpo_lambda}-${dataset_names[$test_dataset_choice]}-answer"
                    fi
                    
                    
                fi

                echo "Start processing $prompt"
                echo "Output to $judge_path"
                echo "Output to $answer_path" 

                python ./response-json-merge.py \
                    --input-dir $answer_path \
                    --output-file $judge_path \

                wait
            done
        done
    done
)