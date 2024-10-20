echo 'if [ -z ${PYTHONPATH+x} ]; then export PYTHONPATH=""; fi' >> /home/root/.bash_profile
echo 'MINTPY_HOME=/home/root/MintPy' >> /home/root/.bash_profile
echo 'PATH=${PATH}:${MINTPY_HOME}/src/mintpy/cli' >> /home/root/.bash_profile
echo 'PYTHONPATH=${PYTHONPATH}:${MINTPY_HOME}/src' >> /home/root/.bash_profile

source /home/root/.bash_profile

bash
