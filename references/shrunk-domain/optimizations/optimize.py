from datetime import datetime
import os
import sys
import torch
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from optimizations.optimization_options import OptimizationOptions
from optimizations.create_optimization import create_optimization


def log_message(message, log_filename):
    """Print and log message

    Args:
        message (str): The message.
        log_filename (str): The log file.

    """
    print(message)
    with open(log_filename, "a") as log_file:
        log_file.write(message)
        log_file.write("\n")


def learn(
    optimization,
    opt,
    writer,
    log_filename,
    log_itr_filename,
    save_label="",
):
    """Optimize by the optimization and the option. Log in writer, log_filename and log_itr_filename.

    Args:
        optimization: The optimization (containing the model).
        opt: The opt, whose training related configs are used.
        writer: The summary writer.
        log_filename (str): The log file for general message.
        log_itr_filename (str): The log file for the last iteration.
        save_label (str): The prefix in saving.

    """
    start = time.time()
    best_objective = 100000
    prev_objective = 100000
    start_itr = opt.start_itr

    # loop over different phases of optimization
    for phase in range(opt.start_phase, len(opt.phase_itrs)):
        optimization.set_optimization(phase=phase)
        optimization.save(label=save_label + "phase_" + str(phase))
        # compute initial objective for logging
        if phase == 0:
            with torch.no_grad():
                optimization.compute_objective(phase=0)
                objective_dict = optimization.get_current_objective()
                for objective_name in objective_dict.keys():
                    writer.add_scalar(
                        tag=objective_name,
                        global_step=opt.start_itr - 1,
                        scalar_value=objective_dict[objective_name],
                    )
                message = "Itr {}, {}".format(opt.start_itr - 1, objective_dict)
                log_message(message, log_filename)
            optimization.save(label=save_label + "initial_force")
        # optimization loop
        for i in range(start_itr, start_itr + opt.phase_itrs[phase]):
            start_i = time.time()
            message, fail = optimization.optimize_parameters(
                phase, is_first_step=(i == opt.start_itr)
            )
            end_i = time.time()

            # print and save to file
            if message is not None:
                message = "Itr {}, ".format(i) + message
                log_message(message, log_filename)
            objective_dict = optimization.get_current_objective()
            for objective_name in objective_dict.keys():
                writer.add_scalar(
                    tag=objective_name,
                    global_step=i,
                    scalar_value=objective_dict[objective_name],
                )

            time_itr = end_i - start_i
            time_total = end_i - start
            if torch.cuda.is_available():
                max_mem = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0
            else:
                max_mem = 0.0

            message = (
                "Itr {}, max memory {:.2f}[GB], elapsed time "
                "{:.0f}[min] {:.1f}[s], total time {:.0f}[min] {:.1f}[s], {}".format(
                    i,
                    max_mem,
                    time_itr // 60,
                    time_itr % 60,
                    time_total // 60,
                    time_total % 60,
                    objective_dict,
                )
            )
            log_message(message, log_filename)

            # save model parameters
            if i % opt.save_param_freq == 0:
                optimization.save(label=save_label + str(i))

            # record best objective and save best model parameters
            if objective_dict["total_objective"] < best_objective:
                best_objective = objective_dict["total_objective"]
                message = "Itr {}, best total objective {}.".format(i, best_objective)
                log_message(message, log_filename)
                optimization.save(label=save_label + "best")

            torch.cuda.empty_cache()

            # end the phase if objective change is very small, only works with smooth objective curve
            if objective_dict["total_objective"] == 0:
                message = (
                    "Phase {} ends at itr {} with total objective reaching 0.".format(
                        phase, i
                    )
                )
                log_message(message, log_filename)
                break
            relative_change = (
                prev_objective - objective_dict["total_objective"]
            ) / prev_objective
            prev_objective = objective_dict["total_objective"]
            if (
                i > start_itr
                and relative_change >= 0
                and relative_change < opt.objective_change_tolerance
            ):
                message = "Phase {} ends at itr {} with relative objective change {:.3f}".format(
                    phase, i, relative_change
                )
                log_message(message, log_filename)
                break
            # end the process if fail switch from optimize_parameters is True, only effective with PyTorch LBFGS
            if fail:
                message = "Phase {} ends at itr {} with line search failure".format(
                    phase, i
                )
                log_message(message, log_filename)
                break

        # note down the last iteration label into log_itr file
        with open(log_itr_filename, "a") as log_itr_file:
            log_itr_file.write("{} ".format(i))

        # save model parameters at the end of the phase
        optimization.save(label=save_label + str(i))

        # advance start_itr to next phase
        start_itr = start_itr + opt.phase_itrs[phase]

    return optimization


def main():
    opt = OptimizationOptions().parse()
    optimization = create_optimization(opt)

    # prepare writer
    writer = SummaryWriter(log_dir=opt.save_path)

    # load model parameters
    if opt.load_param_label is not None:
        optimization.load(label=opt.load_param_label, format="pth")

    # prepare log file and optimiza
    log_filename = os.path.join(opt.save_path, "log.txt")
    with open(log_filename, "w") as log_file:
        log_file.write(
            "Optimization start time: {}\n".format(
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
        )
    log_itr_filename = os.path.join(opt.save_path, "log_itr.txt")
    with open(log_itr_filename, "w") as log_itr_file:
        log_itr_file.write("")
    print("------------ Main Optimization -----------")
    learn(optimization, opt, writer, log_filename, log_itr_filename, save_label="")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
