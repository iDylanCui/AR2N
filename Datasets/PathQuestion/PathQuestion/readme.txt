Here are datasets and their corresponding filtered KB:
PQ-2H(2H-kb)
PQ-3H(3H-kb)
PQL-2H(PQL2-KB)
PQL-3H/PQL-3H_more(PQL3-KB)

Original KB files are:
Freebase13.txt
FB2M_TRI_NEW.txt

======DATA file: 
each line includes: question, answer, answer set, answer path.
each line follows below form:
#for PQ:
question	answer(answer1/answer2/…)	subject#r1#e1#r2#e2..#answer#<end>#answer
#for PQL:
question	answer(answer1/answer2/…)	subject#r1#e1#r2#e2..#answer


=======KB file:
each line includes one fact triple like:
subject		relation	object

本文件夹为原始数据集