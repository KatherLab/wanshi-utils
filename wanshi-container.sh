#!/bin/sh

set -eux

repodir=$(dirname -- "$0")
image_id=$(podman build --target deploy --quiet "$repodir")

podman run --rm -ti \
	--security-opt=label=disable \
	--hooks-dir=/usr/share/containers/oci/hooks.d/ \
	--volume="$HOME":"$HOME" \
	--volume=/mnt:/mnt \
	--volume=/run/media:/run/media \
	--workdir="$PWD" \
	"$image_id" \
	"$@"
