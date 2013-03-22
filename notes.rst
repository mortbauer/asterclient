to run aster here, just make sure the REPE_OUT directory is empty and then fire
up::

    ./asteru Python/Execution/E_SUPERV.py -eficas_path ./Python -commandes \
    fort.1  -num_job 16432 -mode interactif \
    -rep_outils /data/opt/aster/outils \
    -rep_mat /data/opt/aster/STA11.2/materiau -rep_dex \
    /data/opt/aster/STA11.2/datg  -memjeveux 64.0 -tpmax 900
