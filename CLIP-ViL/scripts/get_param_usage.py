import sys

file_prefix = sys.argv[1]

tasks = ["vqa", "gqa", "snli"]

for task in tasks:
    log_name = f"snap/{task}/{file_prefix}/log.log"
    
    try:
        print(task)
        with open(log_name, "r") as f:
            lines = f.readlines()

            for line in lines:
                if "Trainable param percentage" in line:
                    percentage = line.split(" ")[-1]

                    print("percentage: ", percentage)

            print(lines[-3:])

        print("="*10)
    
    except:
        print(f"No {task}")
            