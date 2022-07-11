def create_output(model, tokenizer, train_data, val_data, learning_rate, epochs, batch_size, output_file):
    f = open("../outputs/" + output_file, 'a')
    f.write("## Info\n")
    f.write(f"This was trained on ")
    for i in train_data:
        f.write(f"[{i}](https://github.com/flippe3/fire_2022/tree/master/task_a/data/{i})")

    f.write(f" and validated on [{val_data}](https://github.com/flippe3/fire_2022/tree/master/task_a/data/{val_data})\n\n")

    f.write(f"Model: [{model}](https://huggingface.co/{model})\n\n Tokenizer: [{tokenizer}](https://huggingface.co/{tokenizer})\n\n")
    
    f.write("Hyperparameters:\n")
    f.write(f"- Learning Rate: {learning_rate}\n")
    f.write(f"- Epochs: {epochs}\n")
    f.write(f"- Batch Size: {batch_size}\n")
    f.close()