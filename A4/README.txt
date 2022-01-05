To run the program, please cd to the directory first

To run code for q4
python3 q4.py
or
python q4.py

To run code for q5.1
python3 q5_1.py
or
python q5_1.py

To run code for q5.2
python3 q5_2.py
or
python q5_2.py

in the terminal.

If error ouccurs, try things listed below:
1. Please make sure that libraries like cv2, plt, numpy are up tp date.
2. Please try updating python version to 3.8 or newer.
3. Please make sure that the import module are installed and up to date.
4. Also check if there is a folder called 'Q4' under the same directory and there should be 3 jpg images and 1 mp4 file inside.
5. In q4, mplcursors is the package for selecting points manually. If there is error for this module, try pip install mplcursors
   or pip3 install mplcursors. If still doesn't work you can remove this module and comment out 
   get_points_selected function. If this is the case, line 168 won't work.


Note that:
1. In q4, you can change the case in line 159
2. In q4, you can uncomment the line 168 and comment lines 169 - 179 to select points manually if mplcursors module works.
3. q4.4 will take probably a few miniutes to complete