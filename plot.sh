
DOT="../tmp/treepic.dot"
PNG="../tmp/treepic.png"

a="_1"
b="_2"
c="_3"

python main.py --task plot --load $1 --dot $DOT
dot -T png $DOT$a -o $PNG$a
dot -T png $DOT$b -o $PNG$b
dot -T png $DOT$c -o $PNG$c
