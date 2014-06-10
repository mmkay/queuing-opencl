#/usr/bin/bash
#set -x;

OUTPUT="time-optimized-100threads.csv"

echo 'size;processes;time' > $OUTPUT
mpjhalt machines
mpjboot machines
for size in 100 200 400
do
  cp -f input.txt.$size input.txt
    for processes in 2 3 4 5 9 17 33 65 129
      do
	echo "size $size machines $machines processes $processes"
	TIME=`mpjrun.sh -np $processes -dev niodev -jar target/queuing-0.0.1-SNAPSHOT.jar | grep "Whole computation took" | tr -cd [0-9.]`
	echo $size';'$processes';'$TIME >> $OUTPUT
      done
done
mpjhalt machines
