"""
Copyright (c) 2016 Robosoup
www.robosoup.com

Built with Python 3.5.3 and TensorFlow GPU 1.0.0
"""
import os.path
import re

__RE_BREAK = re.compile("[.?!] ")
__RE_APOSTROPHE = re.compile("[’']")
__RE_DIACRITICS_A = re.compile("[àáâãäå]")
__RE_DIACRITICS_C = re.compile("ç")
__RE_DIACRITICS_E = re.compile("[èéêë]")
__RE_DIACRITICS_I = re.compile("[ìíîï]")
__RE_DIACRITICS_N = re.compile("ñ")
__RE_DIACRITICS_O = re.compile("[òóôõöø]")
__RE_DIACRITICS_U = re.compile("[ùúûü]")
__RE_DIACRITICS_Y = re.compile("[ýÿ]")
__RE_INITIALS = re.compile("([a-z])\.([a-z])")
__RE_NON_ALPHA = re.compile("[^a-z0-9 ]")
__RE_NON_4_DIGIT = re.compile("(^|\s+)[0-9]{1,3}(?=(\s+|$))|(^|\s+)[0-9]{5,}(?=(\s+|$))")
__RE_NON_YEARS = re.compile("(^|\s+)[03-9][0-9]{3}(?=(\s+|$))")
__RE_MULTI_NUM = re.compile("<NUM>\s+(?=<NUM>)")
__RE_NON_A_OR_I = re.compile("(^|\s+)[^ai](?=(\s+|$))")
__RE_MULTI_SPACE = re.compile("\s{2,}")


def __clean_text(text):
    # Convert text to lower case.
    text = text.lower()

    # Remove apostrophes.
    text = __RE_APOSTROPHE.sub("", text)

    # Standardise diacritics.
    text = __RE_DIACRITICS_A.sub("a", text)
    text = __RE_DIACRITICS_C.sub("c", text)
    text = __RE_DIACRITICS_E.sub("e", text)
    text = __RE_DIACRITICS_I.sub("i", text)
    text = __RE_DIACRITICS_N.sub("n", text)
    text = __RE_DIACRITICS_O.sub("o", text)
    text = __RE_DIACRITICS_U.sub("u", text)
    text = __RE_DIACRITICS_Y.sub("y", text)

    # Remove full stops between initials.
    text = __RE_INITIALS.sub("${1}${2}", text)

    # Remove none alpha-numerics and spaces.
    text = __RE_NON_ALPHA.sub(" ", text)

    # Replace free standing numbers(not 4 in length).
    text = __RE_NON_4_DIGIT.sub(" <NUM> ", text)

    # Replace 4 digit numbers that are not years in the range 1000 - 2999.
    text = __RE_NON_YEARS.sub(" <NUM> ", text)

    # Remove repeated numeric markers.
    text = __RE_MULTI_NUM.sub("", text)

    # Remove single free standing letters (not 'a' or 'i').
    text = __RE_NON_A_OR_I.sub("", text)

    # Remove multiple spaces.
    text = __RE_MULTI_SPACE.sub(" ", text)

    # Return clean text.
    return text.strip()


def run(corpus_path, clean_path):
    if not os.path.isfile(clean_path):
        print("creating clean file")
        with open(corpus_path, mode='r', encoding='utf8') as corpus_file:
            with open(clean_path, mode='w') as clean_file:
                counter = 0
                for line in corpus_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    for sub_line in __RE_BREAK.split(line):
                        sub_line = __clean_text(sub_line)
                        if len(sub_line) > 0:
                            clean_file.write(sub_line + "\n")
