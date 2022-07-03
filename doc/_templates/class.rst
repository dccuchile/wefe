:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}


.. autoclass:: {{ objname }}
   :members:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

.. include:: {{module}}.{{objname}}.examples

.. .. raw:: html

..     <div style='clear:both'></div>

.. raw:: html

    <div class="clearer"></div>
