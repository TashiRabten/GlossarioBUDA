#! /usr/bin/python
# encoding: utf-8

from itertools import chain

alphabet = set(['ཀ', 'ཁ', 'ག', 'ང',
                    'ཅ', 'ཆ', 'ཇ', 'ཉ',
                    'ཏ', 'ཐ', 'ད', 'ན',
                    'པ', 'ཕ', 'བ', 'མ',
                    'ཙ', 'ཚ', 'ཛ', 'ཝ',
                    'ཞ', 'ཟ', 'འ', 'ཡ',
                    'ར', 'ལ', 'ཤ', 'ས',
                    'ཧ', 'ཨ'])

pref = set(["ག", "ད", "བ", "མ", "འ"])

head_letter = set(["ར", "ལ", "ས"])

root_only = frozenset(('ཀ', 'ཁ', 'ཅ', 'ཆ', 'ཇ', 'ཉ', 'ཏ', 'ཐ', 'པ', 'ཕ',
                     'ཙ', 'ཚ', 'ཛ', 'ཝ', 'ཞ', 'ཟ', 'ཡ', 'ཤ',))

subcons = frozenset(('ྐ', 'ྑ', 'ྒ', 'ྔ', 'ྕ', 'ྖ', 'ྗ', 'ྙ',
                                        "ྚ", 'ྟ', 'ྠ', 'ྡ', "ྜ", 'ྣ', 'ྤ', 'ྦ',
                                        'ྥ', 'ྨ', 'ྩ', 'ྪ', 'ྫ', 'ྯ', 'ྮ', 'ྴ', 'ྷ', 'ྻ', 'ྼ', 'ྶ'))

subjoined = frozenset(('ྱ', 'ྲ', 'ླ', 'ྭ')) # wazur is being treated as an official member, for now at least

suffixes = set(['ག', 'ང', 'ད', 'ན', 'བ', 'མ', 'འ', 'ར', 'ལ', 'ས'])
second_suffix = set(['ས', 'ད'])
vowels = set(['ི', 'ུ', 'ེ', 'ོ'])

retroflex = frozenset(('ཊ','ཋ','ཌ','ཎ','ཥ',
                      'ྚ', 'ྛ', 'ྜ', 'ྞ', 'ྵ'))


twelve_ra_mgo = set(['རྐ', 'རྒ', 'རྔ', 'རྗ', 'རྙ', 'རྟ', 'རྡ', 'རྣ',
                             'རྦ', 'རྨ', 'རྩ', 'རྫ'])

ten_la_mgo = set(['ལྐ', 'ལྒ', 'ལྔ', 'ལྕ', 'ལྗ', 'ལྟ', 'ལྡ', 'ལྤ',
                        'ལྦ', 'ལྷ'])

eleven_sa_mgo = set(['སྐ', 'སྒ', 'སྔ', 'སྙ', 'སྟ', 'སྡ', 'སྣ', 'སྤ',
                                'སྦ', 'སྨ', 'སྩ'])

wazur_sub = set(['ཀྭ','ཁྭ','གྭ','ཅྭ','ཉྭ','ཏྭ','དྭ','ཙྭ','ཚྭ','ཞྭ','ཟྭ','རྭ',
                 'ལྭ','ཤྭ','སྭ','ཧྭ','གྲྭ', 'དྲྭ'])  #  everything after and including གྲྭ added by me

# 'dogs can combinations
seven_ya_tags = set(['ཀྱ', 'ཁྱ', 'གྱ', 'པྱ', 'ཕྱ', 'བྱ', 'མྱ'])
twelve_ra_tags = set(['ཀྲ', 'ཁྲ', 'གྲ', 'ཏྲ', 'ཐྲ', 'དྲ', 'པྲ', 'ཕྲ', 'བྲ', 
                  'མྲ', 'ཧྲ', 'སྲ'])
six_la_tags = set(['ཀླ', 'གླ', 'བླ', 'ཟླ', 'རླ', 'སླ'])

# three tiered stacks
ya_tags_stack = set(['རྐྱ', 'རྒྱ', 'རྨྱ', 'སྐྱ', 'སྒྱ', 'སྤྱ', 'སྦྱ', 'སྨྱ'])
ra_tags_stack = set(['སྐྲ', 'སྒྲ', 'སྣྲ', 'སྤྲ', 'སྦྲ', 'སྨྲ'])

legal_ga_prefix = frozenset(['གཅ', 'གཉ', 'གཏ', 'གད', 'གན', 'གཙ', 'གཞ', 
                             'གཟ', 'གཡ', 'གཤ', 'གས',])

legal_da_prefix = frozenset(['དཀ', 'དཀྱ', 'དཀྲ', 'དག', 'དགྱ', 'དགྲ', 'དང', 
                            'དཔ', 'དཔྱ', 'དཔྲ', 'དབ', 'དབྱ', 'དབྲ', 'དམ', 
                            'དམྱ',])

legal_ba_prefix = frozenset(['བཀ', 'བཀྱ', 'བཀྲ', 'བརྐ', 'བསྐ', 'བརྐྱ', 'བསྐྱ', 
                             'བསྐྲ', 'བག', 'བགྱ', 'བརྒ', 'བསྒ', 'བརྒྱ', 'བསྒྱ', 
                             'བསྒྲ', 'བརྔ', 'བསྔ', 'བཅ', 'བརྗ', 'བརྙ', 'བསྙ', 
                             'བཏ', 'བརྟ', 'བལྟ', 'བསྟ', 'བད', 'བརྡ', 'བལྡ', 
                             'བསྡ', 'བརྣ', 'བསྣ', 'བཙ', 'བརྩ', 'བསྩ', 'བརྫ', 
                             'བཞ', 'བཟ', 'བཟླ', 'བརླ', 'བཤ', 'བས', 'བསྲ', 
                             'བསླ', 'བགྲ'])

legal_ma_prefix = frozenset(['མཁ', 'མཁྱ', 'མཁྲ', 'མག', 'མགྱ', 'མགྲ', 'མང',
                              'མཆ', 'མཇ', 'མཉ', 'མཐ', 'མད', 'མན', 'མཚ', 
                              'མཛ',])

legal_a_prefix = frozenset(['འཁ','འཁྱ','འཁྲ','འག','འགྱ','འགྲ','འཆ',
                            'འཇ','འཐ','འད','འདྲ','འཕ','འཕྱ','འཕྲ',
                            'འབ','འབྱ','འབྲ','འཚ','འཛ',])

all_legal_prefix = (legal_ga_prefix.union(legal_da_prefix).union(legal_ma_prefix).
                    union(legal_ba_prefix).union(legal_a_prefix))

amb1 = ('བགས', 'མངས')
amb2 = ('དགས', 'འགས', 'དབས', 'དམས')


letters = ('ཀ','ཁ','ག','གྷ','ང','ཅ','ཆ','ཇ','ཉ','ཊ','ཋ','ཌ','ཌྷ','ཎ','ཏ',
           'ཐ','ད','དྷ','ན','པ','ཕ','བ','བྷ','མ','ཙ','ཚ','ཛ','ཛྷ','ཝ',
           'ཞ','ཟ','འ','ཡ','ར','ལ','ཤ','ཥ','ས','ཧ','ཨ','ཀྵ','ཪ','ཫ','ཬ',)

all_stacks = frozenset([i for i in chain(twelve_ra_mgo ,ten_la_mgo , eleven_sa_mgo ,
                       seven_ya_tags , twelve_ra_tags , six_la_tags ,
                       ya_tags_stack , ra_tags_stack , wazur_sub)])

# for achung endings, also consider adding u'འུའོ'
achung_endings = set(['འི', 'འུ', 'འང', 'འམ', 'འོ', 'འུའི'])

sa_yang_endings = set(['གས','ངས','བས','མས',])
da_yang_endings = set(['ནད','རད','ལད',]) # Rare

valid_starts = all_stacks.union(all_legal_prefix).union(alphabet).union(wazur_sub)
valid_endings = suffixes.union(sa_yang_endings).union(achung_endings).union(da_yang_endings)

head = head_letter

hard_indicators = root_only.union(subjoined).union(vowels).union(subcons)

letters = ('ཀ','ཁ','ག','གྷ','ང','ཅ','ཆ','ཇ','ཉ','ཊ','ཋ','ཌ','ཌྷ','ཎ','ཏ',
           'ཐ','ད','དྷ','ན','པ','ཕ','བ','བྷ','མ','ཙ','ཚ','ཛ','ཛྷ','ཝ',
           'ཞ','ཟ','འ','ཡ','ར','ལ','ཤ','ཥ','ས','ཧ','ཨ','ཀྵ','ཪ','ཫ','ཬ',)

subjoined_letters = ('ྐ','ྑ','ྒ','ྒྷ','ྔ','ྕ','ྖ','ྗ','ྙ','ྚ','ྛ','ྜ','ྜྷ',
                      'ྞ','ྟ','ྠ','ྡ','ྡྷ','ྣ','ྤ','ྥ','ྦ','ྦྷ','ྨ','ྩ',
                      'ྪ','ྫ','ྫྷ','ྭ','ྮ','ྯ','ྰ','ྱ','ྲ','ླ','ྴ','ྵ',
                      'ྶ','ྷ','ྸ','ྐྵ','ྺ','ྻ','ྼ',)

f_vowels = ('\\u0f71', '\\u0f72', '\\u0f73', '\\u0f74', '\\u0f75', '\\u0f76',
     '\\u0f77', '\\u0f78', '\\u0f79', '\\u0f7a', '\\u0f7b', '\\u0f7c', '\\u0f7d',
     '\\u0f80', '\\u0f81')

misc_word_parts = ('ྃ', 'ཾ')

word_parts = letters + subjoined_letters + f_vowels + misc_word_parts

def start_end_ok(s, e):
#    if len(vowels.intersection(e)) > 0:
#        return False
    # Wazur in combinations like གྲྭ is not accounted for in main routine
    # This stems from fact wazur is regarded as a subjoined letter
    # even though actually it itself can be appended to a subjoined ltr
    if e and e[0] == 'ྭ' and s:
        s += 'ྭ'
        e = e.lstrip('ྭ')

    if s and s not in valid_starts:
        return False

    # we look at the beginning and ending letters of a word
    # and for the most part assume any vowels in between are behaving
    # as expected. This is not always a good assumption.
    # In this case, we check to make sure there is not more than one
    # consecutive vowel
    if e and e[0] in vowels and len(e) > 1:
        if e[1] in vowels:
            return False

    e = e.lstrip('ིེོུ')

    if e and e not in valid_endings:
        return False

    return True


def find_root_easy(ls):
    '''Hard indicators tell you exactly where root is'''
    root = -1
    for i, l in enumerate(ls):
        
        if len(ls) == 1 and l in alphabet:
            root = 0
            break
        
        elif l in subjoined:

            if not start_end_ok(ls[0:i+1],ls[i+1:]):
#                print start, end
                root = -1
            else: root =  i - 1
            break
        elif l in vowels and ls[i-1] == 'འ' and i-1 != 0:
#            if i-1 != 0:
#                root = -1
#            else:
#                if not start_end_ok(u'འ' , ls[i+1:]):
#                    root = -1
#                else:
#                    root = 0
            if not start_end_ok(ls[:i-1] , ls[i-1:]):
                root = -1
                break
            else:
                root = 0
            break
        elif l in vowels:
            if not start_end_ok(ls[0:i], ls[i:]):
                root = -1
                break
            else: root = i-1
            break
        elif l in root_only or l in subcons:
            try:
                if i+1 <= len(ls) - 1 and ls[i+1] in subjoined:
                    if not start_end_ok(ls[:min(i+2, len(ls))], ls[min(i+2, len(ls)):]):
#                        print ls[:min(i+2, len(ls))], ls[min(i+2, len(ls)):], 'ookl'
                        root = -1
                    else:
                        root = i


                elif not start_end_ok(ls[:min(i+1, len(ls))], ls[min(i+1, len(ls)):]):
#                    print ls[:min(i+1, len(ls))], ls[min(i+1, len(ls)):], 'skldfj'
                    root = -1

                else:
                    root = i
            except IndexError:
                if not start_end_ok(ls, ''):
                     root = -1
                else:
                    root = i
            break


    return root

def find_root_cons(ls):
    '''Find root among a string of non descript consonants'''

    root = -1

    if len(ls) == 1:
        if ls in alphabet:
            root = 0
        else: root = -1

    if len(ls) == 2:
        if ls[0] == ls[1]:
            root = -1
        elif ls[1] == 'འ':
            root = -1
        elif ls[1] not in suffixes:
            root = -1
        else:
            root = 0

    elif len(ls) == 3:
        if ls[-1] in suffixes:
            if (ls[-2:] in sa_yang_endings) or\
               (ls[-2:] in da_yang_endings):
                if ls in amb2: #ambiguous cases
                    root = 1
                else:
                    root = 0
            elif ls[1] == 'འ': # ex བའམ
                if not start_end_ok(ls[0], ls[1:]): root = -1
                else: root = 0
            else:
                if not start_end_ok(ls[0:2], ls[2]): root = -1
                else:
                    root = 1


    elif len(ls) == 4:
#        print 'landed here'
        root = 1
        if not start_end_ok(ls[0:2], ls[2:]): root = -1

    return root


def is_non_std(ls):
    '''Detect whether a group of letters is non standard. ls
    is assumed to be word chars only. i.e. numbers, symbols, etc
    should be removed before calling this function'''

    if not retroflex.isdisjoint(ls):
        return True

    elif not hard_indicators.isdisjoint(ls):
        root_ind = find_root_easy(ls)
        if root_ind not in (0,1,2):
            return True
        else:
            return False

    else:
        if not alphabet.issuperset(ls):
            pass

        root = find_root_cons(ls)
        if root == -1:
            return True
        else:
            return False


def get_root(ls):
    if not retroflex.isdisjoint(ls):
        return ls[0]

    elif not hard_indicators.isdisjoint(ls):
        root_ind = find_root_easy(ls)
        if root_ind not in (0,1,2):
            return ls[0]
        else:
            return ls[root_ind]

    else:
        if not alphabet.issuperset(ls):
#            print 'Warning!', ls
            pass

        root_ind = find_root_cons(ls)
        if root_ind == -1:
            return ls[0]
        else:
            return ls[root_ind]



if __name__ == '__main__':
#    print start_end_ok(u'བ', u'གས')
    samples = 'བཏགས སྒྲུབ པའི འོད སྤྲེའུའི མཏོན  གཏོ ནཔལཐ གཞན མཐའ མདའ བདག ལནག ཀྲ ཁྲ བའམ པའམ མཐའི རེའུ ལ པ བ ན ལྟ རྒྱཔ པོདེ བསྡིག མགྲོད བགྲོད པོའོ  བའི ཧཱུྃ'
    for s in samples.split():
        print((is_non_std(s), s))

    from .termset import syllables
    print(('ཧཱུྃ' in syllables))

