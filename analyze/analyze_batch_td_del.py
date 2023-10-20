import sys
import os
from numpy import mean, std

'''
Usage:
  cat <raw_match_file> | python analyze_batch_td_d.py <test_id>

where 

cat az_test5_cp100_games_5x.txt | python analyze_azlog_gnu.py zz


and output looks like:

Processed 4 games.
final error rates were: [38.3, 91.7, 100.4, 49.9, 102.3, 77.7, 53.2, 56.8]
average error rate:
71.2875


----

'''

'''
On Windows, when writing command files need to look like:
  C:\\Users\\astre\\code\\bg_open_spiel\\my_file.py
For linux you will probably need to modify to something like:
  current_file_path = "/c/Users/astre/code/bg_open_spiel/"

so on windows, e.g.
$ cat ../test_eval.txt | python ../analyze_batch_td_del.py  t1 "C:\\\\Users\\\\astre\\\\code\\\\my_dellalibera_fork\\\\td-gammon\\\\td_gammon\\\\tmp\\\\"
'''

gnubg_name = "gnubg-cli.exe"
#gnubg_name = "gnubg"
#gnubg_name = "/usr/games/gnubg"  # for colab.

# On Linux change above to just
# gnubg_name = "gnubg"


def file_header(match_id):
    header = """
; [Match ID "{0}"]
; [Player 1 "white"]
; [Player 2 "blue"]
; [Variation "Backgammon"]

1 point match

 Game 1
    white : 0                              blue : 0
"""
    return header.format(match_id)

def game_out(game_id, game_str):
    return_str = "\n" + file_header(game_id) + "\n"
    return_str += game_str
    
    return return_str    


def create_gnu_in(subtest_name, current_file_path, match_file_name):
    gnubg_py_in_str =  """
import sys
import gnubg
import sys
import json

"""
    gnubg_py_in_str += 'in_file = \'import auto "' + current_file_path
    gnubg_py_in_str += match_file_name + '" \' ' + "\n"
    
    gnubg_py_in_str +=  """
gnubg.command(in_file)
gnubg.command('analyze match')
gnubg.command('show statistics match')
"""

    temp_py_fn = subtest_name + "_gnubg.py"
    gnubg_py_file = open (current_file_path + temp_py_fn, 'w')
    gnubg_py_file.write (gnubg_py_in_str)
    gnubg_py_file.close ()

    #print ("Created gnubg-cli temporary file: " + temp_py_fn)
    return temp_py_fn

def is_float (str):
    try:
        float(str)
        return True
    except ValueError:
        return False

# calls gnubg in a shell process, extracts analysis of game in string and returns.
def analyze_game(subtest_name, current_file_path, gnubg_match_file_name, gnubg_py_fn):
    gnubg_cmd = gnubg_name + ' -t --python="'
    gnubg_cmd += current_file_path + gnubg_py_fn + '" '
    gnubg_cmd += '| grep mEMG | tail -n 1' # only extract error rates for game.
    # tail is needed b/c 3 error rates are printed but last is for whole game.
    
    print (gnubg_cmd)
    
    stream = os.popen(gnubg_cmd)
    output = stream.read()
    return output

def process_game_str(game_str, current_file_path, test_id, game_id):
    #print "New game_str:"
    #print game_str

    subtest_name = test_id + "_" + str(game_id)
    
    # create file that gnubg can read
    gnubg_match_file_name = subtest_name + "_gnubg.txt"
    gnubg_match_file = open(current_file_path + gnubg_match_file_name, 'w')
    gnubg_match_file.write (game_out(game_id, game_str))
    gnubg_match_file.close()


    print ("Wrote to file " + gnubg_match_file_name)

    # create python file that gnubg will process which analyzes game
    gnubg_py_fn = create_gnu_in(subtest_name, current_file_path,
                                gnubg_match_file_name)
            
    error_rates_str = analyze_game(subtest_name, current_file_path, gnubg_match_file_name, gnubg_py_fn)
    print (error_rates_str)

    print ("error_rates_str: " + error_rates_str)

    # e.g. error_rates_str could be
    # Error rate mEMG (MWC) -53.2   ( -2.662%)      -56.8   ( -2.841%)
    # here we are only interested in the -53.2 and -56.8.
            
    error_rates_split = error_rates_str.split()
    error_rates = []
    for x in error_rates_split:
        unsigned = x.strip('-') # remove negative sign
        if (is_float(unsigned)):
            error_rates.append(float(unsigned)) 
    assert len(error_rates) == 2, "error parsing result of gnubg command"
    return error_rates
        
    
def main():

    usg_msg = """
Usage:
  cat <raw_match_file> | python analyze_azlog_gnu.py test_id current_file_path
where 

raw_match_file is an az log file.
test_id is some identifying string for this run/test.  

example usage 

cat az_test5_cp60_game_3.txt | python convert_to_gnubg_multi.py test_zz <current_file_path>

NOTE: deveoped on Windows, on Linux you will need to modify global variable
current_file_path at top of file.

"""
    
    assert len(sys.argv) == 3, "Missing/incorrect arguments \n" + usg_msg
    
    test_id = sys.argv[1]  # 1st param
    current_file_path = sys.argv[2]
    
    all_error_rates = []
    
    game_id = 0
    game_str = ""
    game_in_progress = False

    for line in sys.stdin:
        
        if game_in_progress:
            if line.strip() == "":
                # new line, end of game
                game_in_progress = False
                error_rates = process_game_str (game_str, current_file_path, test_id, game_id)
                all_error_rates.extend(error_rates)            
                game_id += 1

                game_str = ""
            else:
                game_str += line

        if (line.startswith("as new game:")):
            game_in_progress = True

            


    print ("Processed " + str(game_id) + " games.")
    print ("final error rates were: " + str(all_error_rates))

    # round to 2 decimal places
    
    mean_str = "{:.2f}".format(mean(all_error_rates))
    std_str = "{:.2f}".format(std(all_error_rates))
    
    print ("average error rate and standard deviation: \n" + mean_str + "\t" + std_str)

if __name__ == "__main__":
    main()
