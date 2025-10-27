#cat test8times.sh 
#!/usr/bin/env bash
# launch 8 servers in the background
for i in {0..7}; do
    (   
        echo "Try to start server $is"
        export CUDA_VISIBLE_DEVICES=$i
        port=$(( 29980 + i ))
        python -m sglang.launch_server \
            --model-path /model/path --host 127.0.0.1  --port $port --max-total-tokens 165536 --mem-fraction-static 0.6  --chunked-prefill-size 1024
    ) &
sleep 10
echo "started server"
done
echo "all server started"          # block until all 8 servers are up
echo "to start python test"
# now run your client script 8 times in parallel
for i in {0..7}; do
    echo "turn $i"
    port=$(( 29980 + i ))
    python3 independent_sglang_after_sft.py --use-existing-server --port "$port" &
    sleep 5
done
wait          # wait unt