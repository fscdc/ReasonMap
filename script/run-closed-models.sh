model_list=(
  "Gemini"
  "Doubao-115"
  "Doubao-115-onlytext"
  "Doubao-415"
  "Doubao-415-onlytext"
  "Doubao-428"
  "o3"
  "4o"
)

for model in "${model_list[@]}"; do
    echo "Running model: $model"
    log_name=${model}.txt
    python main_closed_models.py --model "$model" > ./logs/"$log_name"
done