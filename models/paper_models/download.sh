#!/bin/bash
cd ./models/paper_models

wget -O KTH.zip https://umich.box.com/shared/static/dljsf9b5oxn99ymn9f9oe6xky7j5c65o.zip
wget -O S1M.zip https://umich.box.com/shared/static/ejhxyk3r00x59jmvfvv42otwq7kk8l52.zip

unzip KTH.zip
unzip S1M.zip

rm KTH.zip
rm S1M.zip

