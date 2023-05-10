#!/bin/bash
docker kill $(docker ps -q --filter ancestor=scb10x/thaillm)