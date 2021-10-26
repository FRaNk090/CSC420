To run the program, please cd to the directory first. Unpack notMNIST_small.tar first. Then just simply run

python3 a2_code.py 
or 
python a2_code.py

in the terminal.

If error ouccurs, try things listed below:
1. Please make sure that libraries like cv2, plt, numpy are up tp date.
2. Please try updating python version to 3.8 or newer.
3. Please make sure that the import module are installed and up to date.
3. Also check if there is a folder called 'notMNIST_small' under the same directory.


Note that:
1. Tasks are splited into different sections. You can run separate tasks by commenting out other tasks
2. Task 3, 4, 5 use the best learning rate in task 2. If you comment out task 2, by default the learning rate will
   be 0.1 because that's the best learning rate when I run it on my pc. You can also change this if you want.
3. You can reduce the number of epoch to reduce the training time
4. Printed messages are stored in 'Task # print result.txt' where # is the number of task file
5. The result plots are stored under 'plot_result_task#.png'as well.
6. The messages and plots will be different if you run multiple times.