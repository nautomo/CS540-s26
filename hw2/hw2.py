import string
import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    described in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)
    
def shred(filename):
    '''
    This function parses a .txt file into a dict of character counts, where
    the input file may contain any printable ASCII characters and can be short 
    (e.g. a single word) or long (e.g. an article). Ignores case, i.e. merges 
    'A' and 'a' counts together, and so on (this is known as case-folding). 
    Only counts characters A to Z (after case-folding), ignoring all other 
    characters such as space, punctuations, etc.

    Sample Input/Output functionality:
    Input: "Hi! I'll go :-)"
    Output: {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'F': 0,
             'G': 1, 'H': 1, 'I': 2, 'J': 0, 'K': 0, 'L': 2,
             'M': 0, 'N': 0, 'O': 1, 'P': 0, 'Q': 0, 'R': 0,
             'S': 0, 'T': 0, 'U': 0, 'V': 0, 'W': 0, 'X': 0,
             'Y': 0, 'Z': 0}

    Returns: dict of character counts
    '''

    boc=dict()
    with open (filename,encoding='utf-8') as f:
        corpus=f.read()
    f.close()
    #convert all lowercase alphabets to uppercase
    corpus=corpus.upper()
    #initialize X
    for a in string.ascii_uppercase:
        boc[a]=0
    for c in corpus:
        if c in string.ascii_uppercase:
            boc[c]+=1
    return boc



# TODO: add your code here for the assignment
e, s = get_parameter_vectors()
filename = sys.argv[1]
e_prior = float(sys.argv[2])
s_prior = float(sys.argv[3])
letter_counts = shred(filename)
F_english = math.log(e_prior)
F_spanish = math.log(s_prior)

# Q1
count_A = letter_counts["A"]
first_letter_english = count_A * math.log(e[0])
first_letter_spanish = count_A * math.log(s[0])
print("Q1")
print(f"{first_letter_english:.4f}")
print(f"{first_letter_spanish:.4f}")

# Q2
for letter, count in letter_counts.items():
    i = ord(letter)-ord("A")
    if count > 0:
        F_english += count * math.log(e[i])
        F_spanish += count * math.log(s[i])
print("Q2")
print(f"{F_english:.4f}")
print(f"{F_spanish:.4f}")

# Q3
diff = F_spanish-F_english
if diff >= 100:
    posterior_probability = 0
elif diff <= -100:
    posterior_probability = 1
else:
    posterior_probability = 1 / (1 + math.exp(diff))
print("Q3")
print(f"{posterior_probability:.4f}")

# You are free to implement it as you wish!
# Happy Coding!
