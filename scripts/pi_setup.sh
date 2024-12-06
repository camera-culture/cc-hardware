#!/usr/bin/env sh

# Update and install required packages
sudo apt update
sudo apt install -y tmux vim ssh git dhcpcd5 ca-certificates curl

# Remove conflicting Docker packages
for pkg in docker.io docker-doc docker-compose podman-docker containerd runc; do
  sudo apt-get remove -y $pkg
done

# Add Docker's official GPG key and repository
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker $USER
newgrp docker

# Configure static IP
sudo sh -c 'cat <<EOF >> /etc/dhcpcd.conf
interface eth0
nogateway
static ip_address=192.168.2.2/24
static routers=192.168.2.1
static domain_name_servers=192.168.2.1 8.8.8.8

interface wlan0
EOF'

sudo systemctl enable dhcpcd

# Update .bashrc
cat <<'EOF' >> ~/.bashrc

_camera_ssh_keygen() {
    read -p "Enter your ID: " id
    [ -z "$id" ] && { echo "ID is required."; return; }
    read -p "Enter name: " name
    [ -z "$name" ] && { echo "Name is required."; return; }
    read -p "Enter email: " email
    [ -z "$email" ] && { echo "Email is required."; return; }
    read -s -p "Enter passphrase (make sure you remember this): " passphrase; echo
    [ -z "$passphrase" ] && { echo "Passphrase is required."; return; }

    ssh-keygen -f ~/.ssh/id_rsa_$id -N "$passphrase" -C "${name// /_} $email"
    echo "Run the following command to show your public key to be added to GitHub:"
    echo "cat ~/.ssh/id_rsa_$id.pub"
}
alias camera-ssh-keygen='_camera_ssh_keygen'

_camera_ssh_add() {
    read -p "Enter ID: " id
    [ -z "$id" ] && { echo "ID is required."; return; }
    id_rsa="$HOME/.ssh/id_rsa_$id"
    [ ! -f "$id_rsa" ] && { echo "id_rsa file '$id_rsa' doesn't exist. Ensure you have run 'camera-ssh-keygen'."; return; }

    eval "$(ssh-agent -s)"
    ssh-add $id_rsa

    name=$(awk '{print $3}' $id_rsa.pub)
    email=$(awk '{print $4}' $id_rsa.pub)
    name="${name//_/ }"
    export GIT_AUTHOR_NAME="$name"
    export GIT_AUTHOR_EMAIL="$email"
    export GIT_COMMITTER_NAME="$name"
    export GIT_COMMITTER_EMAIL="$email"
}
alias camera-ssh-add="_camera_ssh_add"

alias brc="vi ~/.bashrc"
alias sb="source ~/.bashrc"
alias vi="vim"
EOF
