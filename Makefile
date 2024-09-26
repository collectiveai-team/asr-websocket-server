$(shell touch .env)
include .env
export $(shell sed 's/=.*//' .env)


core-build:
	docker compose build asr_websocket_server-core