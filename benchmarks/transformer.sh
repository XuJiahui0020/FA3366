python transformer_demo.py --label_type next_log --feature_type sequentials --topk 10 --dataset HDFS --data_dir ../data/processed/HDFS/hdfs_0.0_tar

# for lr in 0.001 0.0001;
# do
#     for ba in 1024 2048 4096;
#     do
#         for w in 1 10 100;
#         do
#             python transformer_demo.py \
#                     --learning_rate ${lr} \
#                     --batch_size ${ba} \
#                     --window_size ${w} \
#                     --label_type next_log --feature_type semantics --topk 10 --dataset HDFS \
#                     --data_dir /Users/jiahui/deep-loglizer/data/processed/HDFS_Drain_result01/hdfs_1.0_tar  
#             sleep 10
#         done
#     done
# done
