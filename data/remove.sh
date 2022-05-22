echo deleting pt
for f in MAESTRO/*/*_wav.pt;
    do rm "$f";
done
echo deleting tsv
for f in MAESTRO/*/*.tsv;
    do rm "$f";
done