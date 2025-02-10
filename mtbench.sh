

cd FastChat/fastchat/llm_judge
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bench_list=(mt_bench ultrafeedback capybara alpacafarm)
bench_choice=4
#!/bin/bash
(
    while read line; do
        if [[ $line == Folder:* ]]; then
            folder_path=$(echo $line | cut -d, -f1 | cut -d: -f2-)


            name=$(awk -F/ '{print $NF}' <<< $folder_path)
            model_name=$(awk -F- '{print $1}' <<< $name)
            # echo $model_name
            if [ "$model_name" = "llama" ] || [ "$model_name" = "qwen" ]; then
                dataset_name=$(awk -F- '{print $5}' <<< $name)
            elif [ $model_name = "mistral" ]; then
                dataset_name=$(awk -F- '{print $4}' <<< $name)
            fi
            if [ "$model_name" = "llama" ] || [ "$model_name" = "qwen" ]; then
                method=$(awk -F- '{print $4}' <<< $name)
            elif [ $model_name = "mistral" ]; then
                method=$(awk -F- '{print $3}' <<< $name)
            fi


            if [ $method = "dpo" ]; then
                lambda_value=$(echo "$name" | awk -F '-' '{print $NF}')
                strategy=$(echo "$name" | awk -F '-' '{print $(NF - 1)}')
                echo "model_name: $model_name, dataset_name: $dataset_name, lambda_value: $lambda_value, method: $method"
                if [ "$model_name" = "llama" ]; then
                    bench_loop=("0")
                elif [ $model_name = "mistral" ]; then
                    bench_loop=("3")
                    model_id="mistral/Zephyr-7b-$method-$dataset_name-$strategy-$lambda_value-$bench_list[$bench_choice]"
                elif [ $model_name = "qwen" ]; then
                    bench_loop=("2")
                    model_id="Qwen/Qwen-7b-$method-$dataset_name-$strategy-$lambda_value-$bench_list[$bench_choice]"
                fi
            elif [ $method = "sft" ]; then
                echo "model_name: $model_name, dataset_name: $dataset_name, method: $method"
                if [ "$model_name" = "llama" ]; then
                    bench_loop=("0")
                    folder_path=princeton-nlp/Llama-3-Base-8B-SFT
                elif [ $model_name = "mistral" ]; then
                    bench_loop=("3")
                elif [ $model_name = "qwen" ]; then
                    bench_loop=("2")
                fi
            fi

            
            

            for bench_choice in "${bench_loop[@]}"
            do 
                if [ "$model_name" = "llama" ]; then
                    if [ $method = "dpo" ]; then
                        model_id="meta-llama/Llama-3-8b-$method-$dataset_name-$strategy-$lambda_value-${bench_list[$bench_choice]}"
                    elif [ $method = "sft" ]; then
                        model_id="meta-llama/Llama-3-8b-$dataset_name-sft-${bench_list[$bench_choice]}"
                    fi
                elif [ $model_name = "mistral" ]; then
                    if [ $method = "dpo" ]; then
                        model_id="mistral/Zephyr-7b-$method-$dataset_name-$strategy-$lambda_value-${bench_list[$bench_choice]}"
                    elif [ $method = "sft" ]; then
                        model_id="mistral/Zephyr-7b-$method-$dataset_name-${bench_list[$bench_choice]}"
                    fi
                elif [ $model_name = "qwen" ]; then
                    if [ $method = "dpo" ]; then
                        model_id="Qwen/Qwen-7b-$method-$dataset_name-$strategy-$lambda_value-${bench_list[$bench_choice]}"
                    elif [ $method = "sft" ]; then
                        model_id="Qwen/Qwen-7b-$method-$dataset_name-${bench_list[$bench_choice]}"
                    fi
                fi
                echo $model_id

                python ./gen_model_answer.py \
                    --model-path $folder_path \
                    --model-id $model_id \
                    --bench-name ${bench_list[$bench_choice]} \
                    --num-gpus-total 8 \
                    > gen_answer_print_log.log 2>&1
                wait
            done
        fi
    done < "latest_po_info.txt"
) &
