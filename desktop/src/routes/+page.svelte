<script lang="ts">
    import { invoke } from "@tauri-apps/api/tauri";
    import * as utils from "../components/utils";

    const availableHeightZones = [2, 3, 4, 5, 6];
    const availableWidthZones = [4, 5, 6, 7, 8];

    let nWidthZones = 6;
    let nHeightZones = 4;
    let monitor = "Select monitor";

    $: colourList = [[""]];
    $: isDataReady = false;

    /**
     * Returns a list of available monitors
     */
    async function getMonitors() {
        return ["Monitor 1", "Monitor 2"];
    }

    // async function startUpRoutine() {
    //     let ipcResponse: string = await invoke(
    //         "write_and_wait_for_response_blocking",
    //         { message: "ack_ok" },
    //     );
    //     ipcResponse = ipcResponse.replaceAll("'", "");
    //     ipcResponse.slice(1);
    //     colourList = utils.parseBackendData(ipcResponse);
    //     isDataReady = true;
    //     console.log(colourList);
    // }

    async function startPreview() {
        setInterval(async () => {
            let ipcResponse: string = await invoke(
                "write_and_wait_for_response_blocking",
                { message: "ack_ok" },
            );
            ipcResponse = ipcResponse.replaceAll("'", "");
            ipcResponse.slice(1);
            colourList = utils.parseBackendData(ipcResponse);
            isDataReady = true;
        }, 100);
    }
</script>

<main class="grid place-content-center">
    <button on:click={startPreview}> Connect to server </button>

    <controls class="grid grid-cols-[1fr_2fr] place-content-center">
        <labels class="block text-right p-4">
            <div class="pt-1.5 pb-1">Select monitor</div>
            <div class="pt-1.5 pb-1">Width zones</div>
            <div class="pt-1.5 pb-1">Height zones</div>
        </labels>

        <selectors class="block py-4">
            <monitor-selector class="block text-left">
                {#await getMonitors()}
                    Loading monitors
                {:then monitors}
                    <select
                        bind:value={monitor}
                        class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                    >
                        {#each monitors as each_monitor}
                            <option>{each_monitor}</option>
                        {/each}
                    </select>
                {/await}
            </monitor-selector>

            <hor-selector class="block pt-1.5 pb-1">
                <select
                    bind:value={nWidthZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                >
                    {#each availableWidthZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </hor-selector>

            <vert-selector class="block pt-1 pb-1">
                <select
                    bind:value={nHeightZones}
                    class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-md block px-4 pt-1.5 pb-1"
                >
                    {#each availableHeightZones as each_zone}
                        <option>{each_zone}</option>
                    {/each}
                </select>
            </vert-selector>
        </selectors>
    </controls>

    {#if isDataReady}
        <preview class="block p-4">
            <div class="flex mx-[50px] w-[384px]">
                {#each { length: nWidthZones } as _, i}
                    <width-zone style={colourList[0][i]} />
                {/each}
            </div>
            <div class="flex justify-between">
                <div class="flex flex-col h-[218px]">
                    {#each { length: nHeightZones } as _, i}
                        <height-zone style={colourList[2][i]} />
                    {/each}
                </div>
                <div class="flex flex-col h-[218px]">
                    {#each { length: nHeightZones } as _, i}
                        <height-zone style={colourList[3][i]} />
                    {/each}
                </div>
            </div>
            <div class="flex mx-[50px] w-[384px]">
                {#each { length: nWidthZones } as _, i}
                    <width-zone style={colourList[1][i]} />
                {/each}
            </div>
        </preview>
    {/if}
</main>

<style>
    width-zone {
        @apply w-full h-[50px];
    }

    height-zone {
        @apply w-[50px] h-full;
    }
</style>
