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

# Install Miniforge (ARM-compatible Conda)
CONDA_DIR="$HOME/.conda"
if [ ! -d "$CONDA_DIR" ]; then
  curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh -o ~/miniforge.sh
  bash ~/miniforge.sh -b -p "$CONDA_DIR"
  rm ~/miniforge.sh
  "$CONDA_DIR/bin/conda" init bash
fi

# Disable base environment activation by default
"$CONDA_DIR/bin/conda" config --set auto_activate_base false

# Update .bashrc with project management functions
cat <<'EOF' >> ~/.bashrc

camera_create_proj() {
    project_name="$1"
    if [ -z "$project_name" ]; then
        echo "Project name is required."
        return 1
    fi

    project_dir="$HOME/projects/$project_name"
    mkdir -p "$project_dir"

    # Create alias for the project
    alias "$project_name"="tmux new -A -s $project_name -c $project_dir"

    # Create Conda environment
    if ! "$HOME/.conda/bin/conda" env list | grep -q "$project_name"; then
        "$HOME/.conda/bin/conda" create -y -n "$project_name" python=3.9
        echo "Conda environment '$project_name' created."
    fi

    # Add activation logic to .bashrc
    if ! grep -q "camera_create_proj_logic" ~/.bashrc; then
        cat <<'BRC' >> ~/.bashrc

# camera_create_proj_logic
if [ -n "\$TMUX" ]; then
    session_name=\$(tmux display-message -p '#S')
    if [ -d "\$HOME/projects/\$session_name" ]; then
        source \$HOME/.conda/bin/activate \$session_name 2>/dev/null || true
    fi
fi
BRC
    fi

    echo "Project '$project_name' created at '$project_dir'."
    echo "To use it, run '$project_name' as a command."
}

camera_remove_proj() {
    project_name="$1"
    if [ -z "$project_name" ]; then
        echo "Project name is required."
        return 1
    fi

    project_dir="$HOME/projects/$project_name"

    # Remove project directory
    if [ -d "$project_dir" ]; then
        rm -rf "$project_dir"
        echo "Removed project directory: $project_dir"
    else
        echo "Project directory '$project_dir' does not exist."
    fi

    # Remove alias if it exists
    unalias "$project_name" 2>/dev/null || echo "Alias for '$project_name' does not exist."

    # Remove Conda environment
    if "$HOME/.conda/bin/conda" env list | grep -q "$project_name"; then
        "$HOME/.conda/bin/conda" env remove -y -n "$project_name"
        echo "Removed Conda environment: $project_name"
    fi
}
camera_ssh_keygen() {
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

camera_ssh_add() {
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

alias brc="vi ~/.bashrc"
alias sb="source ~/.bashrc"
alias vi="vim"

# Add custom bashrc below
# ========================

EOF
