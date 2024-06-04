# Reactive layer

Responsible: Oscar Guerrero Rosado (oscar.guerrerorosado@donders.ru.nl)

CAVAA's reactive layer is modeled as a neural mass hypothalamus model that follows attractor dynamics. This model receives internal states as inputs (i.e. energy level, temperature, security, etc.) and imposes a winner-take-all mechanism to define a motivational state that drives goal-oriented behaviors.

In the reactive_layer.py script, you will find two versions of the hypothalamus model.

- The **Multiattractor** class allows the implementation of a variable number of internal needs in the agent, where the number of modeled needs is equal to the number of inputs received. This version was presented at the IROS 2023 workshop Learning robot super autonomy (Detroit, USA). For more information and access to the preprint and full repository check out this [post](https://oscarguerrerorosado.github.io/Post-IROS2023.html).

- The **Attractor** class is a simplified version that allows the implementation of two internal needs in the agent. This version was published as [Guerrrero Rosado, O., Amil, A. F., Freire, I. T., & Verschure, P. F. (2022). Drive competition underlies effective allostatic orchestration. Frontiers in Robotics and AI, 9, 1052998](https://www.frontiersin.org/articles/10.3389/frobt.2022.1052998/full?utm_source=S-TWT&utm_medium=SNET&utm_campaign=ECO_FROBT_XXXXXXXX_auto-dlvrit). A full interactive implementation of this model can be found in the following [Colab notebook](https://colab.research.google.com/drive/1FFLaeUqoFr6AjfKiNdFs76hDWezUlmxY).

