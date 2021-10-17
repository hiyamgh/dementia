#!/usr/bin/env bash

COUNTER="$1"
threshold=2000

echo Input from script is: $COUNTER
for i in {1..5};
    do
        COUNTER=$(expr $COUNTER + 100);
        echo Counter is: $COUNTER
done

if [ $COUNTER -lt $threshold ]; then
    sh ./$0 $COUNTER
    exit 0
fi
