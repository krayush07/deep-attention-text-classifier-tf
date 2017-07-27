import sys

root_folder = '/home/aykumar/aykumar_home/aykumar_dir/multi_view/global_module/utility_dir/vanguard/output/with_resp/90_10_new_architecture_1024_pool_20_5_12345/'

cost_file = open(root_folder + 'test_cost_4neg.txt', 'r')
output = open(root_folder + 'index.txt', 'w')
output_seq = open(root_folder + 'seq_selected.txt', 'w')

test_seq_file = open(root_folder + 'tokenized_test.txt', 'r')



count_iter = 1
min_pos = -1
step_val = 5
pred_ans = ''
pred_goal = ''
pred_slot = ''

min_cost = sys.float_info.max
for costLine, pred_line in zip(cost_file, test_seq_file):
    costLine = costLine.rstrip()
    ans = pred_line.rstrip()

    cost = float(costLine)
    if (count_iter < step_val):
        if (min_cost > cost):
            min_cost = cost
            min_pos = count_iter
            pred_ans = ans
        count_iter += 1
    else:
        if (min_cost > cost):
            min_pos = step_val
            pred_ans = ans
        output.write(str(min_pos) + "\n")
        output_seq.write(pred_ans + "\n")
        count_iter = 1
        min_cost = sys.float_info.max
        min_pos = -1

cost_file.close()
output.close()
output_seq.close()
test_seq_file.close()


output_seq = open(root_folder + 'seq_selected.txt', 'r')
test_act_file = open(root_folder + 'tokenized_test_actual.txt', 'r')
unmatched_file = open(root_folder + 'unmatched.file', 'w')
output_seq_info = open(root_folder + 'seq_selected_with_info.txt', 'w')
correct_incorrect_file = open(root_folder + 'correct_incorrect.txt', 'w')

total = 0.0
correct = 0.0

for l1, l2 in zip(output_seq, test_act_file):
    total += 1
    l1 = l1.rstrip()
    l2 = l2.rstrip()
    if(l1.lower() == l2.lower()):
        correct += 1
        output_seq_info.write(l1 + "\t" + "CORRECT" + "\n")
        correct_incorrect_file.write("correct\n")
    else:
        output_seq_info.write(l1 + "\t" + "INCORRECT" + "\n")
        unmatched_file.write(str(int(total)) + "\t" + l1.split("\t")[2] + "\t" + l2.split("\t")[2] + "\n")
        correct_incorrect_file.write("incorrect\n")

print(correct, total)
output_seq.close()
output_seq_info.close()
test_act_file.close()
unmatched_file.close()
correct_incorrect_file.close()
