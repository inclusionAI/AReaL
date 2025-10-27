import json
import os

def stat(base_directory):
    sum_main_len = 0
    sum_thread_len = 0
    sum_total_len = 0
    total_threads = 0
    total_launches = 0
    count = 0
    for i in range(500):
        
        if i < 10:
            directory = os.path.join(base_directory, f"problem_0{i}")
        else:
            directory = os.path.join(base_directory, f"problem_{i}")
        try:
            with open(os.path.join(directory, "statistics.json"), "r") as f:
                statistics = json.load(f)
        except:
            print(f"problem_{i} not found")
            continue
        
        main_len = statistics["main_answer_length"]
        thread_len = statistics["thread_answers_total_length"]
        number_thread = statistics["total_threads"]
        number_launches = len(statistics["launch_widths"])
        if main_len !=1 :
            thread_len = statistics["thread_answers_total_length"]
            total_len = main_len + thread_len
            count += 1
        # print(f"problem_{i}: {total_len}")
        # print(f"main_len: {main_len}")
        # print(f"thread_len: {thread_len}")
        # print(f"total_len: {total_len}")
        # print("--------------------------------")
            sum_main_len += main_len
            sum_thread_len += thread_len
            sum_total_len += total_len
            total_threads += number_thread
            total_launches += number_launches

        else: print(f"problem_{i} has invalid main length: {main_len}")
    print (f"Total problems processed: {count}")
    print(f"Average main length: {sum_main_len / count}")
    print(f"Average thread length: {sum_thread_len / count}")
    print(f"Average total length: {sum_total_len / count}")
    print(f"Average threads: {total_threads}/ {count}")
    print(f"Average launches: {total_launches}/ {count}")
    print(f"Average threads per launch: {total_threads / total_launches if total_launches > 0 else 0}")

stat("/home/zhangzy/parallel-thinking/Test/terminal_test_24_lr25_nvidia_short_okay_thread/2025-08-20_21-09-53")
