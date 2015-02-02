------------------
Oslo Versioned Objects
------------------

oslo.versionedobjects library deals with DB schema being at a different
version than the code expects, allowing services to be operated safely during
upgrades. It is also used in RPC APIs, to ensure upgrades happen without
spreading version dependant code across different services and projects.

https://launchpad.net/oslo.versionedobjects

For more information, see our wiki page:

   https://wiki.openstack.org/wiki/Oslo

Running Tests
-------------

To run tests in virtualenvs (preferred)::

  sudo pip install tox
  tox
