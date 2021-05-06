fsents = open("linzen_testset/subj_agr_filtered.text", "r")
fwords = open("linzen_testset/subj_agr_filtered.gold", "r")

fout = open("all_data.txt", "w")

for line1, line2 in zip(fsents, fwords):
    line1, line2 = line1.lower(), line2.lower()
    pos, word_gold, word_corrupt, _ = line2.split()
    pos = int(pos)
    line1 = line1.strip().split()
    line1.pop(-1)
    assert line1[pos] == word_gold, (line1, pos, word_gold)
    fout.write(' '.join(line1))
    line1[pos] = word_corrupt
    fout.write('\t' + ' '.join(line1) + '\n')


