#!/bin/bash


curl \
-F "spk=te_m" \
-F "lang=te" \
-F "text=అదే సమయంలో గాలి గదులలో పత్ర రంధ్రాలు, లెంటిసెల్స్  వెలుపల కూడా వాయు సాంద్రతలో తేడా వస్తుంది. అందువల్ల వెలుపలి గాలి పత్ర రంధ్రాలగుండా లోపలికి ప్రవేశిస్తుంది. " \
http://13.233.184.150:5000/predict -o "te_test.wav"
