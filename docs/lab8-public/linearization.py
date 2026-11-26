import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Linearization lab")


@app.cell(hide_code=True)
def models():
    from dataclasses import dataclass

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp


    TOLERANCE = 1e-8
    NONLINEAR_COLOR = "#1765ad"
    CANDIDATE_COLOR = "#58606b"
    SPECIES_COLORS = ("#1765ad", "#9c3970")


    @dataclass(frozen=True)
    class System:
        name: str
        labels: tuple[str, ...]
        guess: tuple[float, ...]
        parameter: tuple[str, float, float, float] | None = None

        def field(self, state, parameter):
            """Also accepts arrays of grid points for drawing the vector field."""
            if self.name == "population":
                x = state[0]
                return np.array([x * (parameter - x)])
            if self.name == "pendulum":
                theta, omega = state
                return np.array([omega, -np.sin(theta) - parameter * omega])
            if self.name == "competition":
                x, y = state
                return np.array([x * (3 - x - 2 * y), y * (2 - x - y)])
            r, theta = state
            return np.array([r * (r - 1), np.sin(theta / 2) ** 2])

        def jacobian(self, state, parameter):
            if self.name == "population":
                return np.array([[parameter - 2 * state[0]]])
            if self.name == "pendulum":
                return np.array([[0.0, 1.0], [-np.cos(state[0]), -parameter]])
            if self.name == "competition":
                x, y = state
                return np.array([[3 - 2 * x - 2 * y, -2 * x], [-y, 2 - x - 2 * y]])
            return np.diag([2 * state[0] - 1, 0.5 * np.sin(state[1])])

        def validate(self, point):
            if not np.all(np.isfinite(point)) or np.max(np.abs(point)) > 1000:
                raise ValueError("Enter finite coordinates between −1000 and 1000.")
            if self.name == "population" and point[0] < 0:
                raise ValueError("Population x must be nonnegative.")
            if self.name == "competition" and np.any(point < 0):
                raise ValueError("Both populations must be nonnegative.")
            if self.name == "polar" and (point[0] < 0 or abs(point[1]) > np.pi):
                raise ValueError("Use r ≥ 0 and −π ≤ θ ≤ π (angles in radians).")


    SYSTEMS = {
        "population": System("population", ("x",), (0.5,), ("P_max", 2, 0.5, 5)),
        "pendulum": System(
            "pendulum", ("θ", "ω"), (0.5, 0.2), ("δ", 0.5, 0, 2)
        ),
        "polar": System("polar", ("r", "θ"), (0.5, 0.5)),
        "competition": System("competition", ("x", "y"), (0.8, 0.8)),
    }


    def equilibrium_check(system, point, parameter):
        residual = system.field(np.asarray(point, dtype=float), parameter)
        return bool(np.max(np.abs(residual)) <= TOLERANCE), residual


    def simulate(field, initial, duration, *, polar=False):
        """Stop before finite-time escape or leaving the lab's angular chart."""
        initial = np.asarray(initial, dtype=float)

        def escape_limit(_time, state):
            return 100 - np.max(np.abs(state))

        escape_limit.terminal = True
        escape_limit.direction = -1
        events = [escape_limit]
        if polar:
            # θ increases here. Do not wrap it or silently extend the lab's chart.
            def chart_boundary(_time, state):
                return np.pi - state[1]

            chart_boundary.terminal = True
            chart_boundary.direction = -1
            events.append(chart_boundary)

        if np.max(np.abs(initial)) >= 100:
            raise ValueError("Use initial coordinates with magnitude below 100.")
        solution = solve_ivp(
            lambda t, y: field(y),
            (0, duration),
            initial,
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
            events=events,
            max_step=0.1,
        )
        end = solution.t[-1]
        times = np.linspace(0, end, 400)
        states = solution.sol(times)
        if not solution.success:
            note = "Integration stopped: " + solution.message
        elif end < duration - 1e-7:
            at_chart_edge = polar and abs(states[1, -1] - np.pi) < 1e-5
            note = (
                "Trajectory stopped at θ = π, the edge of this coordinate chart."
                if at_chart_edge
                else "Trajectory stopped at the safety bound |state| = 100; it is escaping."
            )
        else:
            note = ""
        return times, states, note


    def make_figure(system, point, parameter, duration):
        """Plot the nonlinear trajectory starting at the proposed point."""
        point = np.asarray(point)
        times, states, note = simulate(
            lambda state: system.field(state, parameter), point, duration, polar=system.name == "polar",
        )

        if len(point) == 1:
            fig, time_axis = plt.subplots(figsize=(7, 4), layout="constrained")
            time_axis.plot(times, states[0], color=NONLINEAR_COLOR, label="From proposed point")
            time_axis.set(xlabel="Time t", ylabel="Population x", title="Population over time")
            time_axis.legend(fontsize=8)
        else:
            fig = plt.figure(figsize=(11, 5.2), layout="constrained")
            grid_spec = fig.add_gridspec(2, 2, width_ratios=(1.15, 1))
            phase = fig.add_subplot(grid_spec[:, 0])
            competition = system.name == "competition"
            if competition:
                population_axis = fig.add_subplot(grid_spec[:, 1])
                xlim = (0, max(3.4, point[0] + 0.3))
                ylim = (0, max(2.4, point[1] + 0.3))
            else:
                time_axes = [fig.add_subplot(grid_spec[k, 1]) for k in range(2)]
            if system.name == "polar":
                xlim = (0, max(1.5, point[0] + 0.3))
                ylim = (-np.pi, np.pi)
            elif system.name == "pendulum":
                xlim = (min(-np.pi - 0.4, point[0] - 1), max(np.pi + 0.4, point[0] + 1))
                ylim = (min(-2, point[1] - 0.5), max(2, point[1] + 0.5))
            xx, yy = np.meshgrid(np.linspace(*xlim, 23), np.linspace(*ylim, 23))
            u, v = system.field([xx, yy], parameter)
            magnitude = np.hypot(u, v)
            phase.quiver(xx, yy, u / np.maximum(magnitude, 1e-12),
                         v / np.maximum(magnitude, 1e-12), color="0.72", alpha=0.8,
                         pivot="mid", scale=35)
            phase.plot(*states, color=NONLINEAR_COLOR, label="From proposed point")
            phase.scatter(*point, marker="*", s=150, color=CANDIDATE_COLOR, zorder=6,
                          label="Proposed point")
            phase.set(xlim=xlim, ylim=ylim, xlabel=system.labels[0], ylabel=system.labels[1],
                      title="Phase portrait · arrows show direction")
            phase.legend(fontsize=8, loc="upper left")
            if competition:
                for values, species, color in zip(states, system.labels, SPECIES_COLORS):
                    population_axis.plot(times, values, color=color, label=f"Species {species}")
                phase.set(xlabel="Species x", ylabel="Species y")
                population_axis.set(xlabel="Time t", ylabel="Population (scaled units)",
                                    title="Both species over time")
                population_axis.legend(loc="best")
            else:
                for axis, values, label in zip(time_axes, states, system.labels):
                    axis.plot(times, values, color=NONLINEAR_COLOR)
                    axis.set(xlabel="Time t", ylabel=label)
                time_axes[0].set_title("Coordinates over time")
        for axis in fig.axes:
            axis.grid(alpha=0.18)
        return fig, [note] if note else []

    return SYSTEMS, equilibrium_check, make_figure, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Linearization lab

    1. Calculate a fixed point and its Jacobian, then enter the proposed coordinates.
       Angles are in radians.
    2. Check the equilibrium verdict and the trajectory from your proposed point.
       At a fixed point, the trajectory stays constant.

    The plots update as you change the inputs. See `lab7.md` for the full exercise.
    """)
    return


@app.cell(hide_code=True)
def _(equilibrium_check, make_figure, mo, np, plt):
    def make_controls(system):
        controls = {
            'point': mo.ui.array([
                mo.ui.number(value=value, label=f'Proposed {label}')
                for label, value in zip(system.labels, system.guess)
            ]),
            'duration': mo.ui.slider(0.5, 30, step=0.5, value=8, show_value=True,
                                     label='Time horizon'),
        }
        if system.parameter is not None:
            label, value, low, high = system.parameter
            controls['parameter'] = mo.ui.slider(
                low, high, step=0.1, value=value, show_value=True, label=label,
            )
        return mo.ui.dictionary(controls)

    def show_controls(controls, system):
        sliders = [controls['duration']]
        if system.parameter is not None:
            sliders.insert(0, controls['parameter'])
        return mo.vstack([
            mo.hstack([controls['point'][i] for i in range(len(system.labels))],
                      justify='start', wrap=True),
            mo.hstack(sliders, justify='start', wrap=True),
        ])

    def show_result(system, values):
        point = np.array(values['point'], dtype=float)
        parameter = values.get('parameter')
        try:
            system.validate(point)
            fig, notes = make_figure(system, point, parameter, values['duration'])
        except ValueError as error:
            return mo.callout(mo.md(f'**Check your inputs.** {error}'), kind='warn')

        fixed, _ = equilibrium_check(system, point, parameter)
        verdict = 'This is an equilibrium.' if fixed else 'This is not an equilibrium.'
        output = [mo.callout(
            mo.md(f'**{verdict}**'),
            kind='success' if fixed else 'warn',
        )]
        plt.close(fig)
        output.append(mo.as_html(fig))
        output.extend(mo.md(note) for note in notes)
        return mo.vstack(output)

    return make_controls, show_controls, show_result


@app.cell(hide_code=True)
def _(SYSTEMS, make_controls, mo, show_controls):
    population = make_controls(SYSTEMS['population'])
    mo.vstack([
        mo.md(r"""
        ## 1. Population growth

        $$\dot{x}=x(P_{\max}-x),\qquad x\geq 0,\quad P_{\max}>0.$$

        Find the fixed points and calculate $A=f'(\bar{x})$ at each one.
        """),
        show_controls(population, SYSTEMS['population']),
    ])
    return (population,)


@app.cell(hide_code=True)
def _(SYSTEMS, population, show_result):
    show_result(SYSTEMS['population'], population.value)
    return


@app.cell(hide_code=True)
def _(SYSTEMS, make_controls, mo, show_controls):
    pendulum = make_controls(SYSTEMS['pendulum'])
    mo.vstack([
        mo.md(r"""
        ## 2. Damped pendulum

        $$\ddot{\theta}=-\sin\theta-\delta\dot{\theta}.$$

        Use the state $(\theta,\omega)$, with $\omega=\dot{\theta}$.
        Find the fixed points and their Jacobians.
        """),
        show_controls(pendulum, SYSTEMS['pendulum']),
    ])
    return (pendulum,)


@app.cell(hide_code=True)
def _(SYSTEMS, pendulum, show_result):
    show_result(SYSTEMS['pendulum'], pendulum.value)
    return


@app.cell(hide_code=True)
def _(SYSTEMS, make_controls, mo, show_controls):
    polar = make_controls(SYSTEMS['polar'])
    mo.vstack([
        mo.md(r"""
        ## 3. A system in polar coordinates

        $$
        \left\{\begin{aligned}
        \dot{r} &= r^2-r, \\
        \dot{\theta} &= \sin^2(\theta/2),
        \end{aligned}\right.
        \qquad r\geq 0,\quad -\pi\leq\theta\leq\pi.
        $$

        Find the fixed points in $(r,\theta)$ and their Jacobians.
        """),
        show_controls(polar, SYSTEMS['polar']),
    ])
    return (polar,)


@app.cell(hide_code=True)
def _(SYSTEMS, polar, show_result):
    show_result(SYSTEMS['polar'], polar.value)
    return


@app.cell(hide_code=True)
def _(SYSTEMS, make_controls, mo, show_controls):
    competition = make_controls(SYSTEMS['competition'])
    mo.vstack([
        mo.md(r"""
        ## 4. Competition between two species

        Two species compete for limited resources. Each grows logistically in isolation.
        Competition reduces their growth at different rates.

        $$
        \left\{\begin{aligned}
        \dot{x} &= x(3-x-2y), \\
        \dot{y} &= y(2-x-y),
        \end{aligned}\right.
        \qquad x,y\geq 0.
        $$

        Find the fixed points and their Jacobians. Compare the populations on the same time plot.
        """),
        show_controls(competition, SYSTEMS['competition']),
    ])
    return (competition,)


@app.cell(hide_code=True)
def _(SYSTEMS, competition, show_result):
    show_result(SYSTEMS['competition'], competition.value)
    return


if __name__ == "__main__":
    app.run()
