"""
This script ensures that the specefied network devices are operating as desired.

This includes:
* Are up and configured via NMCLI
* Are using manual IP addresses
* Are using an MTU of 9000
* Finally, that the DHCP Server is running on these adapters

Background: For an unknown reason on the FOPS machine, the 10GBit network
adapters will not persist their nmcli settings.
"""

import nmcli
import click
import logging
import typing
import ipaddress
import subprocess
import time
import tempfile
from pathlib import Path


log = logging.getLogger("nmcli-dhcp-manager")
logging.basicConfig(level=logging.DEBUG)


DHCPD_CONF_STUB = """
subnet {interface} netmask {netmask} {{
 range {range_start} {range_stop};
 option routers {router_ipv4};
}}
"""


@click.command()
@click.option(
    "--devices",
    "-d",
    help="A comma seperated list of devices to manage. E.g. enp3s0,enp4s0",
    required=True,
)
@click.option(
    "--ip-addresses",
    "-ip",
    help="IP address range to use. Each extra adapter will use an incremented /24 range!",
    default="192.168.1.1/24",
    show_default=True,
)
@click.option(
    "--mtu",
    "-m",
    help="MTU to set",
    default=9000,
    show_default=True,
)
def cli(devices: str, ip_addresses: str, mtu: int):
    # Split and sanitise
    input_devices = [
        device_str.lower().strip().rstrip() for device_str in devices.split(",")
    ]
    ip_interface = ipaddress.IPv4Interface(ip_addresses)
    log.debug(f"Attempting to control f{input_devices}")
    log.debug(f"NMCLI devices: {[dev.device for dev in nmcli.device()]}")
    devices_to_control = get_matched_devices(input_devices)
    log.debug(f"Matched devices: {_get_device_str_list(devices_to_control)}")
    connections_to_control = get_matched_connections(devices_to_control)
    log.debug(
        f"Matched connections: {[connection.name for connection in connections_to_control]}"
    )
    ip_interface_current = ip_interface
    device_interface_list = []
    for connection in connections_to_control:
        # A little icky, but generate the next ip address in this network range
        ip_addresses_generator = ip_interface_current.network.hosts()
        ip_address = next(ip_addresses_generator)  # type: ignore
        ip_interface_actual = ipaddress.IPv4Interface(
            f"{ip_address.compressed}/{ip_interface_current.compressed.split('/')[1]}"
        )
        set_static_ip(connection, ip_interface_actual)
        set_mtu(connection, mtu)
        reset_connection(connection)
        device_interface_list.append((connection, ip_interface_actual))
        log.info(f"Reset {connection.name} with {ip_interface_actual}")
        # This is pretty icky, but increments '192.168.1.0/24' -> '192.168.2.0/24'
        ip_interface_current = ipaddress.IPv4Interface(
            f"{(ip_interface_current+255).compressed.split('/')[0]}/{ip_interface_current.compressed.split('/')[1]}"
        )
    log.debug("Waiting 10 seconds for connections to settle")
    time.sleep(10.0)
    restart_dhcp(device_interface_list)


def get_matched_devices(
    devices_str: typing.List[str],
) -> typing.List[nmcli.data.device.Device]:
    matched_devices = []
    for sys_device in nmcli.device():
        if sys_device.device.lower() in devices_str:
            matched_devices.append(sys_device)
    return matched_devices


def get_matched_connections(
    devices: typing.List[nmcli.data.device.Device],
) -> typing.List[nmcli.data.connection.Connection]:
    matched_connections = []
    for sys_connection in nmcli.connection():
        device = sys_connection.device
        # if connection is inactive device == "--"
        if device == "--":
            device = nmcli.connection.show(sys_connection.name)[
                "connection.interface-name"
            ]
        if device in _get_device_str_list(devices):
            matched_connections.append(sys_connection)
    return matched_connections


def _get_device_str_list(
    devices: typing.List[nmcli.data.device.Device],
) -> typing.List[str]:
    return [device.device for device in devices]


def set_static_ip(
    connection: nmcli.data.connection.Connection, ip_interface: ipaddress.IPv4Interface
):
    nmcli.connection.modify(
        connection.name,
        {
            "ipv4.addresses": ip_interface.compressed,
            "ipv4.gateway": ip_interface.ip.compressed,
            "ipv4.method": "manual",
        },
    )


def set_mtu(connection: nmcli.data.connection.Connection, mtu: int):
    nmcli.connection.modify(
        connection.name,
        {"802-3-ethernet.mtu": str(mtu)},
    )


def reset_connection(connection: nmcli.data.connection.Connection):
    try:
        nmcli.connection.down(connection.name, wait=60)
        time.sleep(1.0)
    except Exception as _:
        pass  # happens if device is already down. We mainly care about up.
    nmcli.connection.up(connection.name, wait=60)


def create_new_dhcpd_conf(device_interface_list):
    raise NotImplementedError("Todo")
    file_buffer = ""
    for device, interface in device_interface_list:
        file_buffer += DHCPD_CONF_STUB.format(
            interface="", netmask="", range_start="", range_stop="", router_ipv4=""
        )
    with tempfile.NamedTemporaryFile(mode="wt") as tmpfile:
        tmpfile.write(file_buffer)
        # Sudo replace old file
        command_cp = [
            "sudo",
            "--non-interactive",
            "-E",
            "cp",
            f"{Path(tmpfile.name).resolve()}",
            f"{Path('/etc/dhcp/dhcpd.conf').resolve()}",
        ]
        # subprocess.check_call(command_cp)


def restart_dhcp(device_interface_list):
    """Uses systemctl to restart isc-dhcpd server.

    Note: it is not (yet) in scope to dynamically update the '/etc/dhcp/dhcpd.conf' file."""
    # create_new_dhcpd_conf(device_interface_list)
    command_restart = [
        "sudo",
        "--non-interactive",
        "-E",
        "systemctl",
        "restart",
        "isc-dhcp-server",
    ]
    command_status = [
        "systemctl",
        "show",
        "isc-dhcp-server",
        "--no-page",
    ]
    try:
        subprocess.check_call(command_restart)
        output_str_lines = (
            subprocess.check_output(command_status).decode().splitlines(keepends=False)
        )
        output = {}
        for line in output_str_lines:
            items = line.split("=")
            output.update({items[0]: items[1]})
        if output["ActiveState"] != "active":
            raise ValueError("Failed to get DHCP server running")
    except subprocess.CalledProcessError as exc:
        log.error("Could not restart isc-dhcp-server")
        raise exc


if __name__ == "__main__":
    cli()
