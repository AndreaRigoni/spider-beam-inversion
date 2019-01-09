include $(top_srcdir)/Common.mk


# MODULE = strike_beamfind_n.py
# MODULE = tf_nest.py
MODULE = strike_imgutils.py

export LANGUAGE=$(LANG)

export abs_top_builddir
export abs_top_srcdir

run: ##@beam run module
run: $(MODULE)
	@ python3 $^


define _install_vscode =
 apt-get update; \
 apt install -y software-properties-common apt-transport-https git wget libasound2 pylint numpy; \
 apt install -y libasound2 git; \
 apt install -y python3-tk pylint; \
 wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add -; \
 add-apt-repository -ru "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
 add-apt-repository -u "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"; \
 apt install -y code;
endef


docker-init: USER=root
docker-init: ##@docker install other deps inside docker container
docker-init:
	@ $(_install_vscode)


mdsplus-init: USER=root
mdsplus-init: ##@docker install mdsplus in container
mdsplus-init:
	@ wget http://www.mdsplus.org/dist/mdsplus.gpg.key; \
	  apt-key add mdsplus.gpg.key; \
		add-apt-repository -ru "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
	  add-apt-repository -u "deb http://www.mdsplus.org/dist/Ubuntu16/repo MDSplus Alpha"; \
	  apt install -y mdsplus-alpha; \
		cd /usr/local/mdsplus/mdsobjects/python/; \
		python3 setup.py install;

bash:
	@$@


