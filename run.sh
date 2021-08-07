spark-submit \
	--master yarn \
	--deploy-mode client \
	--num-executors 4 \
	trend.py hdfs://wolf:9000/user/lfq4864/hw2/crime/Crimes_-_2001_to_present.csv hdfs://wolf:9000/user/lfq4864/hw2/out
