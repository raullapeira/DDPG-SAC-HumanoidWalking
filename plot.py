import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv("training_log.csv")

# Media móvil (suavizado)
df["reward_smooth"] = df["reward"].rolling(window=20).mean()

# Crear figura
plt.figure()

# Curva original (ruidosa)
plt.plot(df["step"], df["reward"], alpha=0.3)

# Curva suavizada
plt.plot(df["step"], df["reward_smooth"])

# Labels
plt.xlabel("Steps")
plt.ylabel("Episode Reward")
plt.title("Training Progress")
plt.grid()

# Guardar imagen
plt.savefig("training_plot.png", dpi=300)

# Cerrar figura (importante si ejecutas muchas veces)
plt.close()