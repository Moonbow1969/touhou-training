#!/usr/bin/env bash
set -euo pipefail
umask 077
cd /var/tmp/moonbow-th08-qemu-261127
case_dir=$PWD
image=moonbow-th08-qemu:261127
name=moonbow-th08-qemu-proof-261127
kvm_group=$(stat -c %g /dev/kvm)

test -f evidence/build.done
test -f evidence/download.done
test ! -e vm/proof.qcow2
test ! -e qmp.sock
test ! -e counter.sock
if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Already registered a proof container called $name" >&2
    exit 1
fi

mkdir -m 700 vm/seed
cat > vm/seed/meta-data <<'EOF'
instance-id: moonbow-th08-snapshot-proof-261127
local-hostname: moonbow-snapshot-proof
EOF
{
    cat <<'EOF'
#cloud-config
datasource_list: [NoCloud, None]
disable_root: true
ssh_pwauth: false
package_update: false
package_upgrade: false
write_files:
  - path: /usr/local/lib/moonbow-vm-counter.py
    permissions: '0644'
    content: |
EOF
    sed 's/^/      /' launcher/counter-service.py
    cat <<'EOF'
  - path: /etc/systemd/system/moonbow-vm-counter.service
    permissions: '0644'
    content: |
      [Unit]
      Description=Moonbow private VM snapshot proof counter
      After=local-fs.target
      [Service]
      Type=simple
      ExecStart=/usr/bin/python3 -u /usr/local/lib/moonbow-vm-counter.py
      Restart=on-failure
      [Install]
      WantedBy=multi-user.target
runcmd:
  - [modprobe, virtio_console]
  - [systemctl, daemon-reload]
  - [systemctl, enable, --now, moonbow-vm-counter.service]
EOF
} > vm/seed/user-data

helper=(docker run --rm --network none --user "$(id -u):$(id -g)" --mount "type=bind,src=$case_dir,dst=/case")
"${helper[@]}" --entrypoint xorriso "$image" -as mkisofs -quiet -volid cidata -joliet -rock \
    -output /case/vm/seed.iso /case/vm/seed
"${helper[@]}" --entrypoint qemu-img "$image" create -f qcow2 -F qcow2 \
    -b /case/vm/noble-server-cloudimg-amd64.img /case/vm/proof.qcow2
sha256sum vm/seed.iso launcher/*.py launcher/start-proof.sh > evidence/startup.sha256
start=$(date +%s%N)
printf '%s\n' "$start" > evidence/boot-start-real-ns
docker run -d --name "$name" --init --network none --cpus=2 --memory=2800m --pids-limit=128 \
    --read-only --tmpfs /tmp:mode=1777,size=64m --cap-drop ALL --security-opt no-new-privileges \
    --device /dev/kvm --user "$(id -u):$(id -g)" --group-add "$kvm_group" \
    --mount "type=bind,src=$case_dir,dst=/case" "$image" \
    -machine pc,accel=kvm -cpu host -m 2048 -smp 2 -display none -monitor none -nodefaults \
    -no-reboot -boot order=c -serial file:/case/evidence/guest-console.log \
    -qmp unix:/case/qmp.sock,server=on,wait=off \
    -drive file=/case/vm/proof.qcow2,if=virtio,format=qcow2,cache=none \
    -drive file=/case/vm/seed.iso,media=cdrom,readonly=on,format=raw \
    -netdev user,id=isolated-dhcp,restrict=on \
    -device virtio-net-pci,netdev=isolated-dhcp \
    -device virtio-serial-pci,id=proof-serial \
    -chardev socket,id=proof-channel,path=/case/counter.sock,server=on,wait=off \
    -device virtserialport,chardev=proof-channel,name=org.moonbow.counter > evidence/container-id
docker inspect --format '{{.State.Status}}' "$name"
